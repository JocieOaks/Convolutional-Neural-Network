using ILGPU.Runtime.Cuda;
using ILGPU.Runtime;
using ILGPU;
using Newtonsoft.Json;

public class ConvolutionalKeyLayer : ConvolutionalLayer
{
    public static bool[] Bools { get; set; }
    public static float[] Floats { get; set; }

    [JsonProperty] ColorVector[,] _boolFilterVector;
    [JsonProperty] ColorVector[,] _floatFilterVector;

    public ConvolutionalKeyLayer(int filterSize, int stride, int outputDimensionsMultiplier) : base(filterSize, stride, outputDimensionsMultiplier)
    {
    }

    [JsonConstructor]
    private ConvolutionalKeyLayer() : base()
    {
    }

    public override string Name => "Convolutional Key Layer";

    public override void Backwards(float learningRate)
    {
        base.Backwards(learningRate);

        for (int i = 0; i < _outputDimensions; i++)
        {
            for (int j = 0; j < _filterSize * _filterSize; j++)
            {
                Color gradient = new Color(_filterGradient[i][j * 3], _filterGradient[i][j * 3 + 1], _filterGradient[i][j * 3 + 2]).Clamp(CLAMP);
                for (int k = 0; k < _boolFilterVector[i, j].Length; k++)
                {
                    if (Bools[k])
                        _boolFilterVector[i, j][k] -= learningRate * LEARNINGMULTIPLIER * gradient;
                }

                for(int k = 0; k < _floatFilterVector[i, j].Length; k++)
                {
                    _floatFilterVector[i, j][k] -= learningRate * LEARNINGMULTIPLIER * gradient * Floats[k];
                }

            }
        }
    }

    public override void Forward()
    {
        for(int i = 0; i < _outputDimensions; i++)
        {
            for(int j = 0; j < _filterSize * _filterSize; j++)
            {
                _filters[i][j] = new Color(0);
                for (int k = 0; k < _boolFilterVector[i, j].Length; k++)
                {
                    if (Bools[k])
                        _filters[i][j] += _boolFilterVector[i, j][k];
                }
                for (int k = 0; k < _floatFilterVector[i, j].Length; k++)
                {
                    _filters[i][j] += _floatFilterVector[i, j][k] * Floats[k];
                }
            }
        }

        base.Forward();
    }

    public override void Reset()
    {
        float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize * (Bools.Length + Floats.Length) + _inputDimensions * _filterSize * _filterSize * (Bools.Length + Floats.Length));
        float stdDev = MathF.Sqrt(variance);

        for (int i = 0; i < _outputDimensions; i++)
        {
            for (int j = 0; j < _filterSize * _filterSize; j++)
            {
                for (int k = 0; k < _boolFilterVector[i, j].Length; k++)
                {
                    _boolFilterVector[i, j][k] = Color.RandomGauss(0, stdDev);
                }

                for (int k = 0; k < _floatFilterVector[i, j].Length; k++)
                {
                    _floatFilterVector[i, j][k] = Color.RandomGauss(0, stdDev);
                }
            }
        }
    }

    public override (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] input, FeatureMap[,] outGradients)
    {
        if (_boolFilterVector == null || _floatFilterVector == null)
        {
            BaseStartup(input, outGradients, _dimensionsMultiplier);

            _boolFilterVector = new ColorVector[_outputDimensions, _filterSize * _filterSize];
            _floatFilterVector = new ColorVector[_outputDimensions, _filterSize * _filterSize];

            float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize * (Bools.Length + Floats.Length) + _inputDimensions * _filterSize * _filterSize * (Bools.Length + Floats.Length));
            float stdDev = MathF.Sqrt(variance);

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _filterSize * _filterSize; j++)
                {
                    _boolFilterVector[i, j] = new ColorVector(Bools.Length);
                    for(int k = 0; k < Bools.Length; k++)
                    {
                        _boolFilterVector[i, j][k] = Color.RandomGauss(0, stdDev);
                    }

                    _floatFilterVector[i,j] = new ColorVector(Floats.Length);
                    for (int k = 0; k < Floats.Length; k++)
                    {
                        _floatFilterVector[i, j][k] = Color.RandomGauss(0, stdDev);
                    }
                }
            }
        }
        else
        {
            BaseStartup(input, outGradients, _filters.Length / input.GetLength(0));
        }

        _filters = new Color[_outputDimensions][];
        _filterGradient = new float[_outputDimensions][];

        for (int i = 0; i < _outputDimensions; i++)
        {
            _filters[i] = new Color[_filterSize * _filterSize];
            _filterGradient[i] = new float[_filterSize * _filterSize * 3];
        }

        _deviceInfos = new MemoryBuffer1D<LayerInfo, Stride1D.Dense>[_inputDimensions];
        _deviceFilters = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions];
        _deviceFilterGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_outputDimensions];

        return (_outputs, _inGradients);
    }
}