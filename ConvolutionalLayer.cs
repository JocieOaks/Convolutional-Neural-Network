// See https://aka.ms/new-console-template for more information
#nullable disable

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;

[Serializable]
public class ConvolutionalLayer : Layer
{
    protected const int CLAMP = 1;

    protected const float LEARNINGMULTIPLIER = 1f;

    protected float[][] _kernalGradient;

    [JsonProperty]
    protected Color[][] _kernals;

    private MemoryBuffer1D<LayerInfo, Stride1D.Dense>[] _deviceInfos;
    private MemoryBuffer1D<Color, Stride1D.Dense>[] _deviceKernals;

    public override string Name => "Convolutional Layer";

    public ConvolutionalLayer(int kernalSize, int stride, ref FeatureMap[,] input, int outputDimensionsMultiplier) : base(kernalSize, stride, ref input, outputDimensionsMultiplier)
    {
        if (outputDimensionsMultiplier < 1)
        {
            throw new ArgumentException("Output Dimension Multiplier must be greater than 0. To decrease dimensions, use Fully Connected Layer");
        }

        //Setup kernals and kernal gradients
        _kernals = new Color[_outputDimensions][];
        _kernalGradient = new float[_outputDimensions][];

        float variance = 0.6666f / (_outputDimensions * kernalSize * kernalSize + _inputDimensions * kernalSize * kernalSize);
        float stdDev = MathF.Sqrt(variance);

        for (int i = 0; i < _outputDimensions; i++)
        {
            _kernals[i] = new Color[kernalSize * kernalSize];
            _kernalGradient[i] = new float[kernalSize * kernalSize * 3];
            for (int j = 0; j < kernalSize * kernalSize; j++)
            {
                _kernals[i][j] = Color.RandomGauss(0, stdDev);
            }
        }

        _deviceInfos = new MemoryBuffer1D<LayerInfo, Stride1D.Dense>[_inputDimensions];
        _deviceKernals = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions];
        _deviceKernalGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_outputDimensions];
    }

    protected FeatureMap[,] Convoluted => _outputs;

    public override FeatureMap[,] Backwards(FeatureMap[,] inputs, FeatureMap[,] inGradients, float learningRate)
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
                _deviceInputs[i, j] = inputs[i, j].Allocate(accelerator);
                _deviceOutGradients[i, j] = _outGradients[i, j].AllocateFloat(accelerator);
            }
        }

        for (int i = 0; i < _outputDimensions; i++)
        {
            _deviceKernals[i] = accelerator.Allocate1D(_kernals[i]);
            _deviceKernalGradients[i] = accelerator.Allocate1D<float>(_kernalGradient[i].Length);
            Index3D index = new(Infos(i).OutputWidth, Infos(i).OutputLength, 3);
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInGradients[i, j] = inGradients[i, j].Allocate(accelerator);

                backwardsOutKernal(index, _deviceInGradients[i, j].View, _deviceKernals[i].View, _deviceOutGradients[i % _inputDimensions, j].View, _deviceInfos[i % _inputDimensions].View);
                backwardsGradientKernal(index, _deviceInGradients[i, j].View, _deviceInputs[i % _inputDimensions, j].View, _deviceKernalGradients[i].View, _deviceInfos[i % _inputDimensions].View);
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
            _deviceKernalGradients[i].CopyToCPU(_kernalGradient[i]);
            _deviceKernalGradients[i].Dispose();
            _deviceKernals[i].Dispose();

            for (int j = 0; j < _kernalSize * _kernalSize; j++)
            {
                _kernals[i][j] -= learningRate * LEARNINGMULTIPLIER * new Color(_kernalGradient[i][j * 3], _kernalGradient[i][j * 3 + 1], _kernalGradient[i][j * 3 + 2]).Clamp(CLAMP);
            }

            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInGradients[i, j].Dispose();
            }
        }

        return _outGradients;
    }

    private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceKernalGradients;

    public void BackwardsKernalOnly(FeatureMap[,] inputs, FeatureMap[,] inGradients, float learningRate)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

        var backwardsGradientKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsGradientKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            _deviceInfos[i] = accelerator.Allocate1D(new LayerInfo[] { Infos(i) });
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInputs[i, j] = inputs[i, j].Allocate(accelerator);
            }
        }

        for (int i = 0; i < _outputDimensions; i++)
        {
            _deviceKernalGradients[i] = accelerator.Allocate1D<float>(_kernalGradient[i].Length);
            Index3D index = new(Infos(i).OutputWidth, Infos(i).OutputLength, 3);
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInGradients[i, j] = inGradients[i, j].Allocate(accelerator);

                backwardsGradientKernal(index, _deviceInGradients[i, j].View, _deviceInputs[i % _inputDimensions, j].View, _deviceKernalGradients[i].View, _deviceInfos[i % _inputDimensions].View);
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
            _deviceKernalGradients[i].CopyToCPU(_kernalGradient[i]);
            _deviceKernalGradients[i].Dispose();

            for (int j = 0; j < _kernalSize * _kernalSize; j++)
            {
                _kernals[i][j] -= learningRate * LEARNINGMULTIPLIER * new Color(_kernalGradient[i][j * 3], _kernalGradient[i][j * 3 + 1], _kernalGradient[i][j * 3 + 2]).Clamp(CLAMP);
            }

            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInGradients[i, j].Dispose();
            }
        }
    }

    public override FeatureMap[,] Forward(FeatureMap[,] inputs)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

        var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<LayerInfo>>(ForwardKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            _deviceInfos[i] = accelerator.Allocate1D(new LayerInfo[] { Infos(i) });
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInputs[i, j] = inputs[i, j].Allocate(accelerator);
            }
        }

        for (int i = 0; i < _outputDimensions; i++)
        {
            _deviceKernals[i] = accelerator.Allocate1D(_kernals[i]);
            Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceOutputs[i, j] = _outputs[i, j].AllocateEmpty(accelerator);

                forwardKernal(index, _deviceInputs[i % _inputDimensions, j].View, _deviceOutputs[i, j].View, _deviceKernals[i].View, _deviceInfos[i % _inputDimensions].View);
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
            _deviceKernals[i].Dispose();
        }

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInputs[i, j].Dispose();
            }
            _deviceInfos[i].Dispose();
        }

        return Convoluted;
    }

    protected static void BackwardsKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> kernal, ArrayView<Color> inGradient, ArrayView<float> outGradient, ArrayView<float> kernalGradient, ArrayView<LayerInfo> info)
    {
        float dL = inGradient[info[0].OutputIndex(index.X, index.Y)][index.Z] * info[0].InverseKSquared;

        for (int j = 0; j < info[0].KernalSize; j++)
        {
            for (int i = 0; i < info[0].KernalSize; i++)
            {
                if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                {
                    int kernalIndex = info[0].KernalIndex(i, j);
                    float dK = dL * input[inputIndex][index.Z];
                    Atomic.Add(ref kernalGradient[FloatIndex(kernalIndex, index.Z)], dK);
                    float dP = dL * kernal[kernalIndex][index.Z];
                    Atomic.Add(ref outGradient[FloatIndex(inputIndex, index.Z)], dP);
                }
            }
        }
    }

    protected static void BackwardsOutKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<Color> kernal, ArrayView<float> outGradient, ArrayView<LayerInfo> info)
    {
        float dL = inGradient[info[0].OutputIndex(index.X, index.Y)][index.Z] * info[0].InverseKSquared;

        for (int j = 0; j < info[0].KernalSize; j++)
        {
            for (int i = 0; i < info[0].KernalSize; i++)
            {
                if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                {
                    int kernalIndex = info[0].KernalIndex(i, j);
                    float dP = dL * kernal[kernalIndex][index.Z];
                    Atomic.Add(ref outGradient[FloatIndex(inputIndex, index.Z)], dP);
                }
            }
        }
    }

    protected static void BackwardsGradientKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<Color> input, ArrayView<float> kernalGradient, ArrayView<LayerInfo> info)
    {
        float dL = inGradient[info[0].OutputIndex(index.X, index.Y)][index.Z] * info[0].InverseKSquared;

        for (int j = 0; j < info[0].KernalSize; j++)
        {
            for (int i = 0; i < info[0].KernalSize; i++)
            {
                if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                {
                    int kernalIndex = info[0].KernalIndex(i, j);
                    float dK = dL * input[inputIndex][index.Z];
                    Atomic.Add(ref kernalGradient[FloatIndex(kernalIndex, index.Z)], dK);
                }
            }
        }
    }

    protected static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> convoluted, ArrayView<Color> kernal, ArrayView<LayerInfo> info)
    {
        Color sum = new();

        for (int j = 0; j < info[0].KernalSize; j++)
        {
            for (int i = 0; i < info[0].KernalSize; i++)
            {
                if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                    sum += kernal[info[0].KernalIndex(i, j)] * input[inputIndex];
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