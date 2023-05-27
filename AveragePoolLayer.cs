// See https://aka.ms/new-console-template for more information
using ILGPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime;
using System.Runtime.Serialization;

[Serializable]
public class AveragePoolLayer : Layer
{
    private readonly static float FOURTH = 0.25f;
    float _invK2;
    private readonly FeatureMap[][] _pooled;
    private readonly FeatureMap[][] _dL_dPNext;
    public AveragePoolLayer(int kernalSize, ref FeatureMap[][] input) : base(input.Length, kernalSize, kernalSize)
    {
        _invK2 = 1f / (kernalSize * kernalSize);
        _pooled = new FeatureMap[_dimensions][];
        _dL_dPNext = new FeatureMap[_dimensions][];
        for(int i = 0; i < _dimensions; i++)
        {
            Pad(input[i]);
            _pooled[i] = new FeatureMap[input[i].Length];
            _dL_dPNext[i] = new FeatureMap[input[i].Length];
            for(int j = 0; j < input[i].Length; j++)
            {
                _dL_dPNext[i][j] = new FeatureMap(input[i][j].Width, input[i][j].Length);
                int width = input[i][j].Width / kernalSize;
                int length = input[i][j].Length / kernalSize;
                _pooled[i][j] = new FeatureMap(width, length);
            }
        }
        input = _pooled;
    }

    public AveragePoolLayer() : base(0, 0, 0) { }

    public override FeatureMap[][] Backwards(FeatureMap[][] input, FeatureMap[][] dL_dP, float learningRate)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using (Accelerator accelerator = context.CreateCudaAccelerator(0))
            {
                MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceDL_dPNext = new MemoryBuffer1D<Color, Stride1D.Dense>[_dimensions, dL_dP[0].Length];

                for (int i = 0; i < _dimensions; i++)
                {
                    for (int j = 0; j < dL_dP[i].Length; j++)
                    {
                        deviceDL_dPNext[i, j] = _dL_dPNext[i][j].AllocateEmpty(accelerator);
                        using MemoryBuffer1D<Color, Stride1D.Dense> deviceDL_dP = dL_dP[i][j].Allocate(accelerator);
                        using MemoryBuffer1D<GPUKernalFeatures, Stride1D.Dense> deviceKernalFeatures =
                            accelerator.Allocate1D(new GPUKernalFeatures[] { new GPUKernalFeatures(dL_dP[i][j].Width, _dL_dPNext[i][j].Width, _kernalSize, _stride, _invK2) });

                        Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<GPUKernalFeatures>> forwardKernal =
                            accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<GPUKernalFeatures>>(BackwardsKernal);

                        Index2D index = new(_dL_dPNext[i][j].Width, _dL_dPNext[i][j].Length);

                        forwardKernal(index, deviceDL_dP.View, deviceDL_dPNext[i, j].View, deviceKernalFeatures.View);

                    }
                }

                accelerator.Synchronize();

                for (int i = 0; i < _dimensions; i++)
                {
                    for (int j = 0; j < input[i].Length; j++)
                    {
                        _dL_dPNext[i][j].CopyFromBuffer(deviceDL_dPNext[i, j]);
                        deviceDL_dPNext[i, j].Dispose();
                    }
                }
            }
        }
        return _dL_dPNext;
    }

    public override FeatureMap[][] Forward(FeatureMap[][] input)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using (Accelerator accelerator = context.CreateCudaAccelerator(0))
            {
                MemoryBuffer1D<Color, Stride1D.Dense>[,] devicePooled = new MemoryBuffer1D<Color, Stride1D.Dense>[_dimensions, input[0].Length];

                for (int i = 0; i < _dimensions; i++)
                {
                    for (int j = 0; j < input[i].Length; j++)
                    {
                        devicePooled[i, j] = _pooled[i][j].AllocateEmpty(accelerator);
                        using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = input[i][j].Allocate(accelerator);
                        using MemoryBuffer1D<GPUKernalFeatures, Stride1D.Dense> deviceKernalFeatures =
                            accelerator.Allocate1D(new GPUKernalFeatures[] { new GPUKernalFeatures(input[i][j].Width, _pooled[i][j].Width, _kernalSize, _stride, _invK2) });

                        Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<GPUKernalFeatures>> forwardKernal =
                            accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<GPUKernalFeatures>>(ForwardKernal);

                        Index2D index = new(_pooled[i][j].Width, _pooled[i][j].Length);

                        forwardKernal(index, deviceInput.View, devicePooled[i, j].View, deviceKernalFeatures.View);

                    }
                }

                accelerator.Synchronize();

                for (int i = 0; i < _dimensions; i++)
                {
                    for (int j = 0; j < input[i].Length; j++)
                    {
                        _pooled[i][j].CopyFromBuffer(devicePooled[i, j]);
                        devicePooled[i, j].Dispose();
                    }
                }
            }
        }
        return _pooled;
    }

    [OnDeserialized]
    public void OnDeserialized(StreamingContext context)
    {
        _invK2 = 1f / (_kernalSize * _kernalSize);
    }

    void Backwards(FeatureMap[] dL_dP, FeatureMap[] dL_dPNext)
    {
        for (int i = 0; i < dL_dP.Length; i++)
        {
            Backwards(dL_dP[i], dL_dPNext[i]);
        }
    }

    private void Backwards(FeatureMap dL_dP, FeatureMap dL_dPNext)
    {

        for (int y = 0; y < dL_dPNext.Length; y++)
        {
            for (int x = 0; x < dL_dPNext.Width; x++)
            {
                dL_dPNext[x, y] = dL_dP[x / _kernalSize, y / _kernalSize] * _invK2;
            }
        }
    }

    private static void BackwardsKernal(Index2D index, ArrayView<Color> dL_dP, ArrayView<Color> dL_dPNext, ArrayView<GPUKernalFeatures> kF)
    {
        int position = (index.Y / kF[0].KernalSize) * kF[0].InputWidth  + index.X / kF[0].KernalSize;
        int nextPosition = index.Y * kF[0].OutputWidth + index.X;
        dL_dPNext[nextPosition] = dL_dP[position] * kF[0].InverseKSquared;
    }

    private static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> pooled, ArrayView<GPUKernalFeatures> kF)
    {
        int offset = index.Y * kF[0].Stride * kF[0].InputWidth + index.X * kF[0].Stride;
        Color sum = new Color();
        for(int j = 0; j < kF[0].KernalSize; j++)
        {
            for(int i = 0; i < kF[0].KernalSize; i++)
            {
                sum += input[offset + i];
            }
            offset += kF[0].InputWidth;
        }
        pooled[index.Y * kF[0].OutputWidth + index.X] = sum * kF[0].InverseKSquared;
    }
}
