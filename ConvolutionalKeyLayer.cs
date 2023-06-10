﻿using ILGPU.Runtime.Cuda;
using ILGPU.Runtime;
using ILGPU;
using Newtonsoft.Json;

public class ConvolutionalKeyLayer : Layer, IPrimaryLayer
{
    protected const int CLAMP = 1;

    protected const float LEARNINGMULTIPLIER = 1f;

    private float[,][] _filterGradients;
    private Color[,][] _filters;

    protected MemoryBuffer1D<float, Stride1D.Dense>[,] _deviceFilterGradients;
    protected MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceFilters;
    protected MemoryBuffer1D<LayerInfo, Stride1D.Dense>[] _deviceInfos;

    protected int _dimensionsMultiplier;

    public bool[][] Bools { get; set; }
    public float[][] Floats { get; set; }

    [JsonProperty] ColorVector[,] _boolFilterVector;
    [JsonProperty] ColorVector[,] _floatFilterVector;

    public ConvolutionalKeyLayer(int filterSize, int stride, int outputDimensionsMultiplier) : base(filterSize, stride)
    {
    }

    [JsonConstructor]
    private ConvolutionalKeyLayer() : base()
    {
    }

    public override string Name => "Convolutional Key Layer";

    protected FeatureMap[,] Convoluted => _outputs;

    public override void Backwards(float learningRate)
    {
        Context context = ConvolutionalNeuralNetwork.Context;
        Accelerator accelerator = ConvolutionalNeuralNetwork.Accelerator;

        var backwardsOutKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsOutKernal);
        var backwardsGradientKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsGradientKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            _deviceInfos[i] = accelerator.Allocate1D(new LayerInfo[] { Infos(i) });
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInputs[i, j] = _inputs[i, j].Allocate(accelerator);
                _deviceOutGradients[i, j] = _outGradients[i, j].AllocateFloat(accelerator);
            }
        }

        for (int i = 0; i < _outputDimensions; i++)
        {
            Index3D index = new(Infos(i).OutputWidth, Infos(i).OutputLength, 3);
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceFilters[i, j] = accelerator.Allocate1D(_filters[i, j]);
                _deviceFilterGradients[i, j] = accelerator.Allocate1D<float>(_filterGradients[i, j].Length);
                _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);

                backwardsOutKernal(index, _deviceInGradients[i, j].View, _deviceFilters[i, j].View, _deviceOutGradients[i % _inputDimensions, j].View, _deviceInfos[i % _inputDimensions].View);
                backwardsGradientKernal(index, _deviceInGradients[i, j].View, _deviceInputs[i % _inputDimensions, j].View, _deviceFilterGradients[i, j].View, _deviceInfos[i % _inputDimensions].View);
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
            }
            _deviceInfos[i].Dispose();
        }

        for (int i = 0; i < _outputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInGradients[i, j].Dispose();
                _deviceFilterGradients[i, j].CopyToCPU(_filterGradients[i, j]);
                _deviceFilterGradients[i, j].Dispose();
                _deviceFilters[i, j].Dispose();
            }
        }

        for (int i = 0; i < _outputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                for (int k = 0; k < _filterSize * _filterSize; k++)
                {
                    Color gradient = new Color(_filterGradients[i, j][k * 3], _filterGradients[i, j][k * 3 + 1], _filterGradients[i, j][k * 3 + 2]).Clamp(CLAMP);
                    for (int l = 0; l < _boolFilterVector[i, k].Length; l++)
                    {
                        if (Bools[j][l])
                            _boolFilterVector[i, k][l] -= learningRate * LEARNINGMULTIPLIER * gradient;
                    }

                    for (int l = 0; l < _floatFilterVector[i, k].Length; l++)
                    {
                        _floatFilterVector[i, k][l] -= learningRate * LEARNINGMULTIPLIER * gradient * Floats[j][l];
                    }

                }
            }
        }
    }

    public override void Forward()
    {
        for(int i = 0; i < _outputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                for (int k = 0; k < _filterSize * _filterSize; k++)
                {
                    _filters[i, j][k] = new Color(0);
                    for (int l = 0; l < _boolFilterVector[i, k].Length; l++)
                    {
                        if (Bools[j][l])
                            _filters[i, j][k] += _boolFilterVector[i, k][l];
                    }
                    for (int l = 0; l < _floatFilterVector[i, k].Length; l++)
                    {
                        _filters[i, j][k] += _floatFilterVector[i, k][l] * Floats[j][l];
                    }
                }
            }
        }

        Context context = ConvolutionalNeuralNetwork.Context;
        Accelerator accelerator = ConvolutionalNeuralNetwork.Accelerator;

        var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<LayerInfo>>(ForwardKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            _deviceInfos[i] = accelerator.Allocate1D(new LayerInfo[] { Infos(i) });
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInputs[i, j] = _inputs[i, j].Allocate(accelerator);
            }
        }

        for (int i = 0; i < _outputDimensions; i++)
        {
            
            Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceFilters[i, j] = accelerator.Allocate1D(_filters[i, j]);
                _deviceOutputs[i, j] = Convoluted[i, j].AllocateEmpty(accelerator);

                forwardKernal(index, _deviceInputs[i % _inputDimensions, j].View, _deviceOutputs[i, j].View, _deviceFilters[i, j].View, _deviceInfos[i % _inputDimensions].View);
            }
        }

        accelerator.Synchronize();

        for (int i = 0; i < _outputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                Convoluted[i, j].CopyFromBuffer(_deviceOutputs[i, j]);
                _deviceOutputs[i, j].Dispose();
                _deviceFilters[i, j].Dispose();
            }
            
        }

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInputs[i, j].Dispose();
            }
            _deviceInfos[i].Dispose();
        }
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
            BaseStartup(input, outGradients, _boolFilterVector.GetLength(0) / input.GetLength(0));
        }

        _filters = new Color[_outputDimensions, _batchSize][];
        _filterGradients = new float[_outputDimensions, _batchSize][];

        for (int i = 0; i < _outputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _filters[i, j] = new Color[_filterSize * _filterSize];
                _filterGradients[i, j] = new float[_filterSize * _filterSize * 3];
            }
        }

        _deviceInfos = new MemoryBuffer1D<LayerInfo, Stride1D.Dense>[_inputDimensions];
        _deviceFilters = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions, _batchSize];
        _deviceFilterGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_outputDimensions, _batchSize];

        return (_outputs, _inGradients);
    }

    protected static void BackwardsGradientKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<Color> input, ArrayView<float> filterGradient, ArrayView<LayerInfo> info)
    {
        float dL = inGradient[info[0].OutputIndex(index.X, index.Y)][index.Z] * info[0].InverseKSquared;

        for (int j = 0; j < info[0].FilterSize; j++)
        {
            for (int i = 0; i < info[0].FilterSize; i++)
            {
                if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                {
                    int filterIndex = info[0].FilterIndex(i, j);
                    float dK = dL * input[inputIndex][index.Z];
                    Atomic.Add(ref filterGradient[FloatIndex(filterIndex, index.Z)], dK);
                }
            }
        }
    }

    protected static void BackwardsKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> filter, ArrayView<Color> inGradient, ArrayView<float> outGradient, ArrayView<float> filterGradient, ArrayView<LayerInfo> info)
    {
        float dL = inGradient[info[0].OutputIndex(index.X, index.Y)][index.Z] * info[0].InverseKSquared;

        for (int j = 0; j < info[0].FilterSize; j++)
        {
            for (int i = 0; i < info[0].FilterSize; i++)
            {
                if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                {
                    int filterIndex = info[0].FilterIndex(i, j);
                    float dF = dL * input[inputIndex][index.Z];
                    Atomic.Add(ref filterGradient[FloatIndex(filterIndex, index.Z)], dF);
                    float dP = dL * filter[filterIndex][index.Z];
                    Atomic.Add(ref outGradient[FloatIndex(inputIndex, index.Z)], dP);
                }
            }
        }
    }

    protected static void BackwardsOutKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<Color> filter, ArrayView<float> outGradient, ArrayView<LayerInfo> info)
    {
        float dL = inGradient[info[0].OutputIndex(index.X, index.Y)][index.Z] * info[0].InverseKSquared;

        for (int j = 0; j < info[0].FilterSize; j++)
        {
            for (int i = 0; i < info[0].FilterSize; i++)
            {
                if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                {
                    int filterIndex = info[0].FilterIndex(i, j);
                    float dP = dL * filter[filterIndex][index.Z];
                    Atomic.Add(ref outGradient[FloatIndex(inputIndex, index.Z)], dP);
                }
            }
        }
    }

    protected static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> convoluted, ArrayView<Color> filter, ArrayView<LayerInfo> info)
    {
        Color sum = new();

        for (int j = 0; j < info[0].FilterSize; j++)
        {
            for (int i = 0; i < info[0].FilterSize; i++)
            {
                if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                    sum += filter[info[0].FilterIndex(i, j)] * input[inputIndex];
            }
        }

        convoluted[info[0].OutputIndex(index.X, index.Y)] = sum * info[0].InverseKSquared;
    }

    protected LayerInfo Infos(int index)
    {
        return (LayerInfo)_layerInfos[index % _inputDimensions];
    }

    private static int FloatIndex(int index, int rgb)
    {
        return index * 3 + rgb;
    }
}