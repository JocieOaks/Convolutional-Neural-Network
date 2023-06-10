using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

[Serializable]
public abstract class Layer : ILayer
{
    protected int _batchSize;
    protected MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceInGradients;
    protected MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceInputs;
    protected MemoryBuffer1D<float, Stride1D.Dense>[,] _deviceOutGradients;
    protected MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceOutputs;
    [JsonProperty] protected int _filterSize;
    protected FeatureMap[,] _inGradients;
    protected int _inputDimensions;
    protected FeatureMap[,] _inputs;
    protected ILayerInfo[] _layerInfos;
    protected FeatureMap[,] _outGradients;
    protected int _outputDimensions;
    protected FeatureMap[,] _outputs;
    [JsonProperty] protected int _stride;
    public Layer(int filterSize, int stride)
    {
        _filterSize = filterSize;
        _stride = stride;
    }

    [JsonConstructor]
    protected Layer()
    {
    }

    [JsonIgnore] public abstract string Name { get; }

    [JsonIgnore] public FeatureMap[,] Outputs => _outputs;

    [JsonIgnore] public int OutputDimensions => _outputDimensions;

    public abstract void Backwards(float learningRate);

    public abstract void BackwardsNoUpdate();

    public abstract void Forward();

    public abstract void Reset();

    public abstract (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] inputs, FeatureMap[,] outGradients);

    protected void BaseStartup(FeatureMap[,] inputs, FeatureMap[,] outGradients, int outputDimensionFactor = 1)
    {
        _inputDimensions = inputs.GetLength(0);
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

        _batchSize = inputs.GetLength(1);
        _layerInfos = new ILayerInfo[_inputDimensions];
        _inputs = inputs;
        _outGradients = outGradients;
        _outputs = new FeatureMap[_outputDimensions, _batchSize];
        _inGradients = new FeatureMap[_outputDimensions, _batchSize];
        
        for (int i = 0; i < _inputDimensions; i++)
        {
            ILayerInfo layer;
            if (_stride == 1 && _filterSize == 1)
            {
                layer = _layerInfos[i] = new SingleLayerInfo()
                {
                    Width = inputs[i, 0].Width,
                    Length = inputs[i, 0].Length,
                };
            }
            else
            {
                layer = _layerInfos[i] = new LayerInfo()
                {
                    FilterSize = _filterSize,
                    Stride = _stride,
                    InverseKSquared = 1f / (_filterSize * _filterSize),
                    InputWidth = inputs[i, 0].Width,
                    InputLength = inputs[i, 0].Length,
                    OutputWidth = 2 + (inputs[i, 0].Width - _filterSize - 1) / _stride,
                    OutputLength = 2 + (inputs[i, 0].Length - _filterSize - 1) / _stride
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
    }
}