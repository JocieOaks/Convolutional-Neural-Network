using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

[Serializable]
public abstract class Layer : ILayer
{
    [JsonProperty] protected int _filterSize;
    [JsonProperty] protected int _stride;
    protected int _inputDimensions;
    protected int _outputDimensions;
    protected int _batchSize;

    protected FeatureMap[,] _outputs;
    protected FeatureMap[,] _outGradients;
    protected ILayerInfo[] _layerInfos;

    protected MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceInputs;
    protected MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceInGradients;
    protected MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceOutputs;
    protected MemoryBuffer1D<float, Stride1D.Dense>[,] _deviceOutGradients;

    [JsonIgnore] public abstract string Name { get; }

    public Layer(int filterSize, int stride)
    {
        _filterSize = filterSize;
        _stride = stride;
    }

    [JsonConstructor]
    protected Layer()
    {
    }

    public abstract FeatureMap[,] Startup(FeatureMap[,] input);

    protected FeatureMap[,] BaseStartup(FeatureMap[,] input, int outputDimensionFactor = 1)
    {
        _inputDimensions = input.GetLength(0);
        if (outputDimensionFactor >= 1)
        {
            _outputDimensions = outputDimensionFactor * _inputDimensions;
        }
        else
        {
            if (outputDimensionFactor == 0 || _inputDimensions % outputDimensionFactor != 0)
            {
                throw new ArgumentException("outputDimensionFactor does not divide evenly with input dimensions.");
            }
            else
            {
                _outputDimensions = _inputDimensions / -outputDimensionFactor;
            }
        }

        _batchSize = input.GetLength(1);
        _layerInfos = new ILayerInfo[_inputDimensions];
        _outGradients = new FeatureMap[_inputDimensions, _batchSize];
        _outputs = new FeatureMap[_outputDimensions, _batchSize];
        for (int i = 0; i < _inputDimensions; i++)
        {
            ILayerInfo layer;
            if (_stride == 1 && _filterSize == 1)
            {
                layer = _layerInfos[i] = new SingleLayerInfo()
                {
                    Width = input[i, 0].Width,
                    Length = input[i, 0].Length,
                };
            }
            else
            {
                layer = _layerInfos[i] = new LayerInfo()
                {
                    FilterSize = _filterSize,
                    Stride = _stride,
                    InverseKSquared = 1f / (_filterSize * _filterSize),
                    InputWidth = input[i, 0].Width,
                    InputLength = input[i, 0].Length,
                    OutputWidth = 2 + (input[i, 0].Width - _filterSize - 1) / _stride,
                    OutputLength = 2 + (input[i, 0].Length - _filterSize - 1) / _stride
                };
            }

            for (int j = 0; j < _batchSize; j++)
            {
                _outGradients[i, j] = new FeatureMap(layer.InputWidth, layer.InputLength);
            }
        }
        for (int i = 0; i < _outputDimensions; i++)
        {
            ILayerInfo layer;
            if (outputDimensionFactor >= 1)
            {
                layer = _layerInfos[i / outputDimensionFactor];
            }
            else
            {
                layer = _layerInfos[i * -outputDimensionFactor];
            }

            for (int j = 0; j < _batchSize; j++)
            {
                _outputs[i, j] = new FeatureMap(layer.OutputWidth, layer.OutputLength);
            }
        }

        _deviceInputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];
        _deviceInGradients = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions, _batchSize];
        _deviceOutputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions, _batchSize];
        _deviceOutGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions, _batchSize];

        return _outputs;
    }

    public abstract FeatureMap[,] Forward(FeatureMap[,] input);

    public abstract FeatureMap[,] Backwards(FeatureMap[,] input, FeatureMap[,] inGradient, float learningRate);
}