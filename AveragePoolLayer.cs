// See https://aka.ms/new-console-template for more information
using ILGPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime;
using System.Runtime.Serialization;

[Serializable]
public class AveragePoolLayer : Layer
{
    float _invK2;
    private FeatureMap[,] Pooled => _outputs;
    public AveragePoolLayer(int kernalSize, ref FeatureMap[,] input) : base(kernalSize, kernalSize, ref input)
    {
    }

    public override FeatureMap[,] Backwards(FeatureMap[,] input, FeatureMap[,] inGradient, float learningRate)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using (Accelerator accelerator = context.CreateCudaAccelerator(0))
            {
                MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceOutGradient = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, inGradient.GetLength(1)];

                for (int i = 0; i < _inputDimensions; i++)
                {
                    for (int j = 0; j < inGradient.GetLength(1); j++)
                    {
                        deviceOutGradient[i, j] = _outGradients[i, j].AllocateEmpty(accelerator);
                        using MemoryBuffer1D<Color, Stride1D.Dense> deviceInGradient = inGradient[i, j].Allocate(accelerator);
                        using MemoryBuffer1D<LayerInfo, Stride1D.Dense> deviceLayerInfo =
                            accelerator.Allocate1D(new LayerInfo[] { _layerInfos[i] });

                        Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<LayerInfo>> forwardKernal =
                            accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<LayerInfo>>(BackwardsKernal);

                        Index2D index = new(_outGradients[i,j].Width, _outGradients[i,j].Length);

                        forwardKernal(index, deviceInGradient.View, deviceOutGradient[i, j].View, deviceLayerInfo.View);

                    }
                }

                accelerator.Synchronize();

                for (int i = 0; i < _inputDimensions; i++)
                {
                    for (int j = 0; j < input.GetLength(1); j++)
                    {
                        _outGradients[i,j].CopyFromBuffer(deviceOutGradient[i, j]);
                        deviceOutGradient[i, j].Dispose();
                    }
                }
            }
        }
        return _outGradients;
    }

    public override FeatureMap[,] Forward(FeatureMap[,] input)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using (Accelerator accelerator = context.CreateCudaAccelerator(0))
            {
                MemoryBuffer1D<Color, Stride1D.Dense>[,] devicePooled = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, input.GetLength(1)];

                for (int i = 0; i < _inputDimensions; i++)
                {
                    for (int j = 0; j < input.GetLength(1); j++)
                    {
                        devicePooled[i, j] = Pooled[i, j].AllocateEmpty(accelerator);
                        using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = input[i,j].Allocate(accelerator);
                        using MemoryBuffer1D<LayerInfo, Stride1D.Dense> deviceLayerInfo =
                            accelerator.Allocate1D(new LayerInfo[] { _layerInfos[i] });

                        Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<LayerInfo>> forwardKernal =
                            accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<LayerInfo>>(ForwardKernal);

                        Index2D index = new(Pooled[i,j].Width, Pooled[i,j].Length);

                        forwardKernal(index, deviceInput.View, devicePooled[i, j].View, deviceLayerInfo.View);

                    }
                }

                accelerator.Synchronize();

                for (int i = 0; i < _inputDimensions; i++)
                {
                    for (int j = 0; j < input.GetLength(1); j++)
                    {
                        Pooled[i,j].CopyFromBuffer(devicePooled[i, j]);
                        devicePooled[i, j].Dispose();
                    }
                }
            }
        }
        return Pooled;
    }

    [OnDeserialized]
    public void OnDeserialized(StreamingContext context)
    {
        _invK2 = 1f / (_kernalSize * _kernalSize);
    }

    private static void BackwardsKernal(Index2D index, ArrayView<Color> inGradient, ArrayView<Color> outGradient, ArrayView<LayerInfo> layer)
    {
        //Unlike other Backwards Kernals, this kernal indexes by the outGradient rather than the inGradient, so the equations for index are inverted.
        int inGradientIndex = (index.Y / layer[0].KernalSize) * layer[0].OutputWidth  + index.X / layer[0].KernalSize;
        int outGradientIndex = index.Y * layer[0].InputWidth + index.X;
        outGradient[outGradientIndex] = inGradient[inGradientIndex] * layer[0].InverseKSquared;
    }

    private static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> pooled, ArrayView<LayerInfo> info)
    {
        Color sum = new();
        for(int j = 0; j < info[0].KernalSize; j++)
        {
            for(int i = 0; i < info[0].KernalSize; i++)
            {
                if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                    sum += input[inputIndex];
            }
        }
        pooled[info[0].OutputIndex(index.X, index.Y)] = sum * info[0].InverseKSquared;
    }
}
