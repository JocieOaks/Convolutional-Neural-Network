
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;

[Serializable]
public class AveragePoolLayer : Layer, IStructuralLayer
{
    private FeatureMap[,] Pooled => _outputs;

    private MemoryBuffer1D<LayerInfo, Stride1D.Dense>[] _deviceInfos;

    public AveragePoolLayer(int filterSize) : base(filterSize, filterSize)
    {
    }

    [JsonConstructor]
    private AveragePoolLayer() : base()
    {
    }

    public override (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] input, FeatureMap[,] outGradients)
    {
        BaseStartup(input, outGradients);
        _deviceInfos = new MemoryBuffer1D<LayerInfo, Stride1D.Dense>[_inputDimensions];
        return (_outputs, _inGradients);
    }

    [JsonIgnore] public override string Name => "Average Pool Layer";

    public override void Backwards(float learningRate)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

        var backwardsKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            _deviceInfos[i] = accelerator.Allocate1D(new LayerInfo[] { Infos(i) });
            Index3D index = new(Infos(i).InputWidth, Infos(i).InputLength, 3);
            for (int j = 0; j < _inGradients.GetLength(1); j++)
            {
                _deviceOutGradients[i, j] = _outGradients[i, j].AllocateFloat(accelerator);
                _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);

                backwardsKernal(index, _deviceInGradients[i, j].View, _deviceOutGradients[i, j].View, _deviceInfos[i].View);
            }
        }

        accelerator.Synchronize();

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _outGradients[i, j].CopyFromBuffer(_deviceOutGradients[i, j]);
                _deviceOutGradients[i, j].Dispose();
                _deviceInGradients[i, j].Dispose();
            }
            _deviceInfos[i].Dispose();
        }
    }

    public override void Forward()
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

        var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<LayerInfo>>(ForwardKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            _deviceInfos[i] = accelerator.Allocate1D(new LayerInfo[] { Infos(i) });
            for (int j = 0; j < _inputs.GetLength(1); j++)
            {
                _deviceOutputs[i, j] = Pooled[i, j].AllocateEmpty(accelerator);
                _deviceInputs[i, j] = _inputs[i, j].Allocate(accelerator);

                Index2D index = new(Pooled[i, j].Width, Pooled[i, j].Length);

                forwardKernal(index, _deviceInputs[i, j].View, _deviceOutputs[i, j].View, _deviceInfos[i].View);
            }
        }

        accelerator.Synchronize();

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _inputs.GetLength(1); j++)
            {
                Pooled[i, j].CopyFromBuffer(_deviceOutputs[i, j]);
                _deviceOutputs[i, j].Dispose();
                _deviceInputs[i, j].Dispose();
            }
            _deviceInfos[i].Dispose();
        }
    }

    private LayerInfo Infos(int index)
    {
        return (LayerInfo)_layerInfos[index];
    }

    private static void BackwardsKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<float> outGradient, ArrayView<LayerInfo> layer)
    {
        //Unlike other Backwards Kernals, this kernal indexes by the outGradient rather than the inGradient, so the equations for index are inverted.
        int inGradientIndex = (index.Y / layer[0].FilterSize) * layer[0].OutputWidth + index.X / layer[0].FilterSize;
        int outGradientIndex = index.Y * layer[0].InputWidth + index.X;
        outGradient[3 * outGradientIndex + index.Z] = inGradient[inGradientIndex][index.Z] * layer[0].InverseKSquared;
    }

    private static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> pooled, ArrayView<LayerInfo> info)
    {
        Color sum = new();
        for (int j = 0; j < info[0].FilterSize; j++)
        {
            for (int i = 0; i < info[0].FilterSize; i++)
            {
                if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                    sum += input[inputIndex];
            }
        }
        pooled[info[0].OutputIndex(index.X, index.Y)] = sum * info[0].InverseKSquared;
    }

    public override void Reset()
    {
    }
}