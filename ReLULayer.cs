using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

public class ReLULayer : Layer
{
    public ReLULayer(ref FeatureMap[,] input) : base(1, 1, ref input) { }

    public override string Name => "Activation Layer";

    public override FeatureMap[,] Backwards(FeatureMap[,] input, FeatureMap[,] inGradient, float learningRate)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

        MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceOutGradients = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];
        for (int i = 0; i < _inputDimensions; i++)
        {
            using var deviceInfo = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });
            Index2D index = new(Infos(i).Width, Infos(i).Length);
            for (int j = 0; j < _batchSize; j++)
            {
                deviceOutGradients[i, j] = input[i, j].AllocateEmpty(accelerator);
                using var deviceInput = input[i, j].Allocate(accelerator);
                using var deviceInGradient = inGradient[i, j].Allocate(accelerator);

                var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<SingleLayerInfo>>(BackwardsKernal);

                forwardKernal(index, deviceInput.View, deviceInGradient.View, deviceOutGradients[i, j].View, deviceInfo.View);
            }
        }

        accelerator.Synchronize();

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _outGradients[i, j].CopyFromBuffer(deviceOutGradients[i, j]);
                deviceOutGradients[i, j].Dispose();
            }
        }

        return _outGradients;
    }

    public override FeatureMap[,] Forward(FeatureMap[,] input)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

        MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceOutputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];
        for (int i = 0; i < _inputDimensions; i++)
        {
            using var deviceInfo = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });
            Index2D index = new(Infos(i).Width, Infos(i).Length);
            for (int j = 0; j < _batchSize; j++)
            {
                deviceOutputs[i, j] = input[i, j].AllocateEmpty(accelerator);
                using var deviceInput = input[i, j].Allocate(accelerator);

                var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<SingleLayerInfo>>(ForwardKernal);

                forwardKernal(index, deviceInput.View, deviceOutputs[i,j].View, deviceInfo.View);
            }
        }

        accelerator.Synchronize();

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _outputs[i, j].CopyFromBuffer(deviceOutputs[i,j]);
                deviceOutputs[i, j].Dispose();
            }
        }

        return _outputs;
    }

    private static void BackwardsKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> inGradient, ArrayView<Color> outGradient, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        outGradient[mapsIndex] = input[mapsIndex].ReLUPropogation() * inGradient[mapsIndex];
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

