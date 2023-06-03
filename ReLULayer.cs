using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;

#nullable disable

[Serializable]
public class ReLULayer : Layer
{
    private MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense>[] _deviceInfos;

    public ReLULayer(ref FeatureMap[,] input) : base(1, 1)
    {
        input = Startup(input);
    }

    [JsonConstructor] private ReLULayer() : base() { }

    public override FeatureMap[,] Startup(FeatureMap[,] input, int outputDimensionFactor = 0)
    {
        BaseStartup(input);
        _deviceInfos = new MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense>[_inputDimensions];
        return _outputs;
    }

    public override string Name => "Activation Layer";

    public override FeatureMap[,] Backwards(FeatureMap[,] input, FeatureMap[,] inGradient, float learningRate)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

        var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(BackwardsKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            _deviceInfos[i] = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });
            Index3D index = new(Infos(i).Width, Infos(i).Length, 3);
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceOutGradients[i, j] = input[i, j].AllocateFloat(accelerator);
                _deviceInputs[i, j] = input[i, j].Allocate(accelerator);
                _deviceInGradients[i, j] = inGradient[i, j].Allocate(accelerator);

                forwardKernal(index, _deviceInputs[i, j].View, _deviceInGradients[i, j].View, _deviceOutGradients[i, j].View, _deviceInfos[i].View);
            }
        }

        accelerator.Synchronize();

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _outGradients[i, j].CopyFromBuffer(_deviceOutGradients[i, j]);

                _deviceOutGradients[i, j].Dispose();
                _deviceInputs[i, j].Dispose();
                _deviceInGradients[i, j].Dispose();
            }

            _deviceInfos[i].Dispose();
        }

        return _outGradients;
    }

    public override FeatureMap[,] Forward(FeatureMap[,] input)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

        var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<SingleLayerInfo>>(ForwardKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            _deviceInfos[i] = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });
            Index2D index = new(Infos(i).Width, Infos(i).Length);

            for (int j = 0; j < _batchSize; j++)
            {
                _deviceOutputs[i, j] = input[i, j].AllocateEmpty(accelerator);
                _deviceInputs[i, j] = input[i, j].Allocate(accelerator);

                forwardKernal(index, _deviceInputs[i, j].View, _deviceOutputs[i, j].View, _deviceInfos[i].View);
            }
        }

        accelerator.Synchronize();

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _outputs[i, j].CopyFromBuffer(_deviceOutputs[i, j]);

                _deviceOutputs[i, j].Dispose();
                _deviceInputs[i, j].Dispose();
            }

            _deviceInfos[i].Dispose();
        }

        return _outputs;
    }

    private static void BackwardsKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> inGradient, ArrayView<float> outGradient, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        outGradient[3 * mapsIndex + index.Z] = input[mapsIndex].ReLUPropogation()[index.Z] * inGradient[mapsIndex][index.Z];
    }

    private static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> output, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        output[mapsIndex] = input[mapsIndex].ReLU();
    }

    private SingleLayerInfo Infos(int index)
    {
        return (SingleLayerInfo)_layerInfos[index];
    }
}