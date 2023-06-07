using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;

public class DropoutLayer : Layer, ISecondaryLayer
{
    private MemoryBuffer1D<int, Stride1D.Dense>[] _deviceDropout;
    private int[][] _dropout;
    [JsonProperty] private float _dropoutRate = 0.2f;

    public DropoutLayer(float dropoutRate) : base(1, 1)
    {
        _dropoutRate = dropoutRate;
    }

    [JsonConstructor]
    private DropoutLayer() : base(1, 1)
    {
    }

    public override string Name => "Dropout Layer";

    public override FeatureMap[,] Backwards(FeatureMap[,] input, FeatureMap[,] inGradient, float learningRate)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);
        var backwardsKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<int>, ArrayView<float>>(BackwardsKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            Index2D index = new(Infos(i).Area, 3);
            _deviceDropout[i] = accelerator.Allocate1D(_dropout[i]);
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInGradients[i, j] = inGradient[i, j].Allocate(accelerator);
                _deviceOutGradients[i, j] = _outGradients[i, j].AllocateFloat(accelerator);
                backwardsKernal(index, _deviceInGradients[i, j].View, _deviceDropout[i].View, _deviceOutGradients[i, j].View);
            }
        }

        accelerator.Synchronize();

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _outGradients[i, j].CopyFromBuffer(_deviceOutGradients[i, j]);
                _deviceInGradients[i, j].Dispose();
                _deviceOutGradients[i, j].Dispose();
            }
            _deviceDropout[i].Dispose();
        }

        return _outGradients;
    }

    public override FeatureMap[,] Forward(FeatureMap[,] input)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);
        var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Color>, ArrayView<int>, ArrayView<Color>>(ForwardKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _dropout[i].Length; j++)
            {
                _dropout[i][j] = ConvolutionalNeuralNetwork.Random.NextDouble() < _dropoutRate ? 0 : 1;
            }
            Index1D index = new Index1D(Infos(i).Area);
            _deviceDropout[i] = accelerator.Allocate1D(_dropout[i]);
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInputs[i, j] = input[i, j].Allocate(accelerator);
                _deviceOutputs[i, j] = _outputs[i, j].AllocateEmpty(accelerator);
                forwardKernal(index, _deviceInputs[i, j].View, _deviceDropout[i].View, _deviceOutputs[i, j].View);
            }
        }

        accelerator.Synchronize();

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _outputs[i, j].CopyFromBuffer(_deviceOutputs[i, j]);
                _deviceInputs[i, j].Dispose();
                _deviceOutputs[i, j].Dispose();
            }
            _deviceDropout[i].Dispose();
        }

        return _outputs;
    }

    public FeatureMap[,] ForwardInference(FeatureMap[,] input)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);
        var inferenceKernal = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Color>, ArrayView<float>, ArrayView<Color>>(InferenceKernal);

        MemoryBuffer1D<float, Stride1D.Dense> deviceRatio = accelerator.Allocate1D(new float[] { 1 - _dropoutRate });
        for (int i = 0; i < _inputDimensions; i++)
        {
            Index1D index = new Index1D(Infos(i).Area);
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInputs[i, j] = input[i, j].Allocate(accelerator);
                _deviceOutputs[i, j] = _outputs[i, j].AllocateEmpty(accelerator);
                inferenceKernal(index, _deviceInputs[i, j].View, deviceRatio.View, _deviceOutputs[i, j].View);
            }
        }

        accelerator.Synchronize();

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _outputs[i, j].CopyFromBuffer(_deviceOutputs[i, j]);
                _deviceInputs[i, j].Dispose();
                _deviceOutputs[i, j].Dispose();
            }
        }

        return _outputs;
    }

    public override FeatureMap[,] Startup(FeatureMap[,] input)
    {
        BaseStartup(input);

        _dropout = new int[_inputDimensions][];
        for (int i = 0; i < _inputDimensions; i++)
        {
            _dropout[i] = new int[input[i, 0].Area * 3];
        }
        _deviceDropout = new MemoryBuffer1D<int, Stride1D.Dense>[_inputDimensions];

        return _outputs;
    }

    private static void BackwardsKernal(Index2D index, ArrayView<Color> inGradient, ArrayView<int> dropout, ArrayView<float> outGradient)
    {
        int arrayIndex = index.X * 3 + index.Y;
        outGradient[arrayIndex] = dropout[arrayIndex] == 0 ? 0 : inGradient[index.X][index.Y];
    }

    private static void ForwardKernal(Index1D index, ArrayView<Color> input, ArrayView<int> dropout, ArrayView<Color> output)
    {
        float r = dropout[3 * index] == 0 ? 0 :input[index].R;
        float g = dropout[3 * index + 1] == 0 ? 0 : input[index].G;
        float b = dropout[3 * index + 2] == 0 ? 0 :input[index].B;
        output[index] = new Color(r, g, b);
    }

    private static void InferenceKernal(Index1D index, ArrayView<Color> input, ArrayView<float> dropoutRate, ArrayView<Color> output)
    {
        output[index] = input[index] * dropoutRate[0];
    }

    private SingleLayerInfo Infos(int index)
    {
        return (SingleLayerInfo)_layerInfos[index];
    }

    public override void Reset()
    {
    }
}