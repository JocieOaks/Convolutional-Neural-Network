using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;

[Serializable]
public class ConvolutionalLayer : Layer, IPrimaryLayer
{
    protected const int CLAMP = 1;

    protected const float LEARNINGMULTIPLIER = 1f;

    protected float[][] _filterGradient;

    [JsonProperty] protected Color[][] _filters;

    private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceFilterGradients;
    private MemoryBuffer1D<Color, Stride1D.Dense>[] _deviceFilters;
    private MemoryBuffer1D<LayerInfo, Stride1D.Dense>[] _deviceInfos;

    private int _dimensionsMultiplier;

    public ConvolutionalLayer(int filterSize, int stride, int outputDimensionsMultiplier) : base(filterSize, stride)
    {
        _dimensionsMultiplier = outputDimensionsMultiplier;
    }

    [JsonConstructor]
    private ConvolutionalLayer() : base()
    {
    }

    public override string Name => "Convolutional Layer";

    protected FeatureMap[,] Convoluted => _outputs;

    public override void Backwards(float learningRate)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

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
            _deviceFilters[i] = accelerator.Allocate1D(_filters[i]);
            _deviceFilterGradients[i] = accelerator.Allocate1D<float>(_filterGradient[i].Length);
            Index3D index = new(Infos(i).OutputWidth, Infos(i).OutputLength, 3);
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);

                backwardsOutKernal(index, _deviceInGradients[i, j].View, _deviceFilters[i].View, _deviceOutGradients[i % _inputDimensions, j].View, _deviceInfos[i % _inputDimensions].View);
                backwardsGradientKernal(index, _deviceInGradients[i, j].View, _deviceInputs[i % _inputDimensions, j].View, _deviceFilterGradients[i].View, _deviceInfos[i % _inputDimensions].View);
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
            _deviceFilterGradients[i].CopyToCPU(_filterGradient[i]);
            _deviceFilterGradients[i].Dispose();
            _deviceFilters[i].Dispose();

            for (int j = 0; j < _filterSize * _filterSize; j++)
            {
                _filters[i][j] -= learningRate * LEARNINGMULTIPLIER * new Color(_filterGradient[i][j * 3], _filterGradient[i][j * 3 + 1], _filterGradient[i][j * 3 + 2]).Clamp(CLAMP);
            }

            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInGradients[i, j].Dispose();
            }
        }
    }

    public void BackwardsFilterOnly(float learningRate)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

        var backwardsGradientKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsGradientKernal);

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
            _deviceFilterGradients[i] = accelerator.Allocate1D<float>(_filterGradient[i].Length);
            Index3D index = new(Infos(i).OutputWidth, Infos(i).OutputLength, 3);
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);

                backwardsGradientKernal(index, _deviceInGradients[i, j].View, _deviceInputs[i % _inputDimensions, j].View, _deviceFilterGradients[i].View, _deviceInfos[i % _inputDimensions].View);
            }
        }

        accelerator.Synchronize();

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInputs[i, j].Dispose();
            }
            _deviceInfos[i].Dispose();
        }

        for (int i = 0; i < _outputDimensions; i++)
        {
            _deviceFilterGradients[i].CopyToCPU(_filterGradient[i]);
            _deviceFilterGradients[i].Dispose();

            for (int j = 0; j < _filterSize * _filterSize; j++)
            {
                _filters[i][j] -= learningRate * LEARNINGMULTIPLIER * new Color(_filterGradient[i][j * 3], _filterGradient[i][j * 3 + 1], _filterGradient[i][j * 3 + 2]).Clamp(CLAMP);
            }

            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInGradients[i, j].Dispose();
            }
        }
    }

    public override void Forward()
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

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
            _deviceFilters[i] = accelerator.Allocate1D(_filters[i]);
            Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceOutputs[i, j] = Convoluted[i, j].AllocateEmpty(accelerator);

                forwardKernal(index, _deviceInputs[i % _inputDimensions, j].View, _deviceOutputs[i, j].View, _deviceFilters[i].View, _deviceInfos[i % _inputDimensions].View);
            }
        }

        accelerator.Synchronize();

        for (int i = 0; i < _outputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                Convoluted[i, j].CopyFromBuffer(_deviceOutputs[i, j]);
                _deviceOutputs[i, j].Dispose();
            }
            _deviceFilters[i].Dispose();
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
        float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize + _inputDimensions * _filterSize * _filterSize);
        float stdDev = MathF.Sqrt(variance);

        for (int i = 0; i < _outputDimensions; i++)
        {
            for (int j = 0; j < _filterSize * _filterSize; j++)
            {
                _filters[i][j] = Color.RandomGauss(0, stdDev);
            }
        }
    }

    public override (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] input, FeatureMap[,] outGradients)
    {
        if (_filters == null)
        {
            BaseStartup(input, outGradients, _dimensionsMultiplier);
            _filters = new Color[_outputDimensions][];

            float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize + _inputDimensions * _filterSize * _filterSize);
            float stdDev = MathF.Sqrt(variance);

            for (int i = 0; i < _outputDimensions; i++)
            {
                _filters[i] = new Color[_filterSize * _filterSize];
                for (int j = 0; j < _filterSize * _filterSize; j++)
                {
                    _filters[i][j] = Color.RandomGauss(0, stdDev);
                }
            }
        }
        else
        {
            BaseStartup(input, outGradients, _filters.Length / input.GetLength(0));
        }
        _filterGradient = new float[_outputDimensions][];

        for (int i = 0; i < _outputDimensions; i++)
        {
            _filterGradient[i] = new float[_filterSize * _filterSize * 3];
        }

        _deviceInfos = new MemoryBuffer1D<LayerInfo, Stride1D.Dense>[_inputDimensions];
        _deviceFilters = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions];
        _deviceFilterGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_outputDimensions];

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