using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;

[Serializable]
public class FullyConnectedLayer : Layer, IPrimaryLayer
{
    private MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense>[] _deviceInfos;
    private MemoryBuffer1D<float, Stride1D.Dense>[,] _deviceMultiplierGradients;
    private MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceMultipliers;
    private new MemoryBuffer1D<float, Stride1D.Dense>[,] _deviceOutputs;

    [JsonProperty] private FeatureMap _matrixBlue;
    [JsonProperty] private FeatureMap _matrixGreen;
    [JsonProperty] private FeatureMap _matrixRed;

    public FullyConnectedLayer(int outputDimensions) : base(1, 1)
    {
        _outputDimensions = outputDimensions;
    }

    [JsonConstructor]
    private FullyConnectedLayer() : base()
    {
    }

    public override FeatureMap[,] Startup(FeatureMap[,] input)
    {
        if (_matrixRed == null)
        {
            BaseStartup(input, -input.GetLength(0) / _outputDimensions);
            float variance = 0.666f / (_inputDimensions + _outputDimensions);
            float stdDev = MathF.Sqrt(variance);
            _matrixRed = new FeatureMap(_inputDimensions, _outputDimensions);
            _matrixGreen = new FeatureMap(_inputDimensions, _outputDimensions);
            _matrixBlue = new FeatureMap(_inputDimensions, _outputDimensions);
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _outputDimensions; j++)
                {
                    _matrixRed[i, j] = Color.RandomGauss(0, stdDev);
                    _matrixGreen[i, j] = Color.RandomGauss(0, stdDev);
                    _matrixBlue[i, j] = Color.RandomGauss(0, stdDev);
                }
            }
        }
        else
        {
            BaseStartup(input, -_matrixRed.Width / _matrixRed.Length);
        }
        _deviceInfos = new MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense>[_inputDimensions];
        _deviceMultiplierGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions, _outputDimensions];
        _deviceMultipliers = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _outputDimensions];
        _deviceOutputs = new MemoryBuffer1D<float, Stride1D.Dense>[_outputDimensions, _batchSize];
        return _outputs;
    }

    public override string Name => "Fully Connected Layer";

    public override FeatureMap[,] Backwards(FeatureMap[,] inputs, FeatureMap[,] inGradients, float learningRate)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

        var backwardsOutKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(BackwardsOutKernal);

        var backwardsGradientKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(BackwardsGradientKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            _deviceInfos[i] = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceOutGradients[i, j] = _outGradients[i, j].AllocateFloat(accelerator);
                _deviceInputs[i, j] = inputs[i, j].Allocate(accelerator);
            }
        }

        for (int i = 0; i < _outputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInGradients[i, j] = inGradients[i, j].Allocate(accelerator);
            }
        }

        for (int i = 0; i < _inputDimensions; i++)
        {
            Index3D index = new(Infos(i).Width, Infos(i).Length, 3);
            for (int j = 0; j < _outputDimensions; j++)
            {
                _deviceMultipliers[i, j] = accelerator.Allocate1D(new Color[] { _matrixRed[i, j], _matrixGreen[i, j], _matrixBlue[i, j] });
                _deviceMultiplierGradients[i, j] = accelerator.Allocate1D<float>(9);
                for (int k = 0; k < _batchSize; k++)
                {
                    backwardsOutKernal(index, _deviceInGradients[j, k].View, _deviceMultipliers[i, j].View, _deviceOutGradients[i, k].View, _deviceInfos[i].View);
                    backwardsGradientKernal(index, _deviceInGradients[j, k].View, _deviceInputs[i, k].View, _deviceMultiplierGradients[i, j].View, _deviceInfos[i].View);
                }
            }
        }

        accelerator.Synchronize();

        for (int j = 0; j < _batchSize; j++)
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                _outGradients[i, j].CopyFromBuffer(_deviceOutGradients[i, j]);

                _deviceOutGradients[i, j].Dispose();
                _deviceInputs[i, j].Dispose();
            }
            for (int i = 0; i < _outputDimensions; i++)
            {
                _deviceInGradients[i, j].Dispose();
            }
        }

        float[] multiplierGradients = new float[9];

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _outputDimensions; j++)
            {
                _deviceMultiplierGradients[i, j].CopyToCPU(multiplierGradients);
                _deviceMultiplierGradients[i, j].Dispose();
                _deviceMultipliers[i, j].Dispose();

                _matrixRed[i, j] -= new Color(multiplierGradients[0], multiplierGradients[1], multiplierGradients[2]) * learningRate;
                _matrixGreen[i, j] -= new Color(multiplierGradients[3], multiplierGradients[4], multiplierGradients[5]) * learningRate;
                _matrixBlue[i, j] -= new Color(multiplierGradients[6], multiplierGradients[7], multiplierGradients[8]) * learningRate;
            }

            _deviceInfos[i].Dispose();
        }

        return _outGradients;
    }

    public override FeatureMap[,] Forward(FeatureMap[,] inputs)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

        var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<Color>, ArrayView<SingleLayerInfo>>(ForwardKernal);

        for (int i = 0; i < _outputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _deviceOutputs[i, j] = _outputs[i, j].AllocateFloat(accelerator);
            }
        }

        for (int i = 0; i < _inputDimensions; i++)
        {
            _deviceInfos[i] = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });

            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInputs[i, j] = inputs[i, j].Allocate(accelerator);
            }
        }

        for (int i = 0; i < _inputDimensions; i++)
        {
            Index3D index = new(Infos(i).Width, Infos(i).Length, 3);
            for (int j = 0; j < _outputDimensions; j++)
            {
                _deviceMultipliers[i, j] = accelerator.Allocate1D(new Color[] { _matrixRed[i, j], _matrixGreen[i, j], _matrixBlue[i, j] });

                for (int k = 0; k < _batchSize; k++)
                {
                    forwardKernal(index, _deviceInputs[i, k].View, _deviceOutputs[j, k].View, _deviceMultipliers[i, j].View, _deviceInfos[i].View);
                }
            }
        }

        accelerator.Synchronize();

        for (int j = 0; j < _batchSize; j++)
        {
            for (int i = 0; i < _outputDimensions; i++)
            {
                _outputs[i, j].CopyFromBuffer(_deviceOutputs[i, j]);

                _deviceOutputs[i, j].Dispose();
            }

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInputs[i, j].Dispose();
            }
        }

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _outputDimensions; j++)
            {
                _deviceMultipliers[i, j].Dispose();
            }
            _deviceInfos[i].Dispose();
        }

        return _outputs;
    }

    private static void BackwardsGradientKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<Color> input, ArrayView<float> multiplierGradient, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        for (int i = 0; i < 3; i++)
        {
            Atomic.Add(ref multiplierGradient[index.Z * 3 + i], inGradient[mapsIndex][index.Z] * input[mapsIndex][i]);
        }
    }

    private static void BackwardsOutKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<Color> multiplier, ArrayView<float> outGradient, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        float transposeDot = 0;
        for (int i = 0; i < 3; i++)
        {
            transposeDot += inGradient[mapsIndex][i] * multiplier[i][index.Z];
        }
        Atomic.Add(ref outGradient[mapsIndex * 3 + index.Z], transposeDot);
    }

    private static void ForwardKernal(Index3D index, ArrayView<Color> input, ArrayView<float> output, ArrayView<Color> multiplier, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        Atomic.Add(ref output[mapsIndex * 3 + index.Z], Color.Dot(input[mapsIndex], multiplier[index.Z]));
    }

    private SingleLayerInfo Infos(int index)
    {
        return (SingleLayerInfo)_layerInfos[index];
    }

    public override void Reset()
    {
        float variance = 0.666f / (_inputDimensions + _outputDimensions);
        float stdDev = MathF.Sqrt(variance);
        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _outputDimensions; j++)
            {
                _matrixRed[i, j] = Color.RandomGauss(0, stdDev);
                _matrixGreen[i, j] = Color.RandomGauss(0, stdDev);
                _matrixBlue[i, j] = Color.RandomGauss(0, stdDev);
            }
        }
    }
}