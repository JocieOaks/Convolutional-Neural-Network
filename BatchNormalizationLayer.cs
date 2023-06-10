using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;
using System.Runtime.InteropServices;

[Serializable]
public class BatchNormalizationLayer : Layer, ISecondaryLayer
{
    [JsonProperty] private ColorVector _bias;
    [JsonProperty] private ColorVector _weight;

    private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceGradients;
    private MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense>[] _deviceInfos;
    private MemoryBuffer1D<Color, Stride1D.Dense>[] _deviceMeans;
    private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceSums;
    private MemoryBuffer1D<Color, Stride1D.Dense>[] _deviceValues;
    private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceVariances;

    private ColorVector _mean;
    private ColorVector _sigma;

    [JsonConstructor]
    public BatchNormalizationLayer() : base(1, 1)
    {
    }

    [JsonIgnore] public override string Name => "Batch Normalization Layer";

    private FeatureMap[,] Normalized => _outputs;

    public override void Backwards(float learningRate)
    {
        Context context = ConvolutionalNeuralNetwork.Context;
        Accelerator accelerator = ConvolutionalNeuralNetwork.Accelerator;

        Gradients[] gradients = new Gradients[_inputDimensions];

        var gradientKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(GradientsKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            _deviceInfos[i] = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });
            _deviceGradients[i] = accelerator.Allocate1D<float>(9);
            _deviceMeans[i] = accelerator.Allocate1D(new Color[] { _mean[i] });
            Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInputs[i, j] = _inputs[i, j].Allocate(accelerator);
                _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);
                _deviceOutputs[i, j] = Normalized[i, j].Allocate(accelerator);

                gradientKernal(index, _deviceInputs[i, j].View, _deviceInGradients[i, j].View, _deviceOutputs[i, j].View, _deviceMeans[i].View, _deviceGradients[i].View, _deviceInfos[i].View);
            }
        }

        accelerator.Synchronize();

        var backwardsKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<Color>, ArrayView<SingleLayerInfo>>(BackwardsKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            gradients[i] = new();
            gradients[i].CopyFromBuffer(_deviceGradients[i]);
            _deviceGradients[i].Dispose();

            gradients[i].SigmaGradient *= Color.Pow(_sigma[i], -1.5f) * _weight[i] * -0.5f;
            gradients[i].MeanGradient = -gradients[i].BiasGradient * _weight[i] / _sigma[i];

            float invM = 1f / (Infos(i).Area * _batchSize);

            _deviceValues[i] = accelerator.Allocate1D(new Color[] { _weight[i] / _sigma[i], 2 * invM * gradients[i].SigmaGradient, _mean[i], invM * gradients[i].MeanGradient });

            Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

            for (int j = 0; j < _batchSize; j++)
            {
                _deviceOutGradients[i, j] = _outGradients[i, j].AllocateFloat(accelerator);

                backwardsKernal(index, _deviceInputs[i, j].View, _deviceInGradients[i, j].View, _deviceOutGradients[i, j].View, _deviceValues[i].View, _deviceInfos[i].View);
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
                _deviceOutputs[i, j].Dispose();
            }

            _weight[i] -= learningRate * gradients[i].WeightGradient.Clamp(1);
            _bias[i] -= learningRate * gradients[i].BiasGradient.Clamp(1);

            _deviceInfos[i].Dispose();
            _deviceMeans[i].Dispose();
            _deviceValues[i].Dispose();
        }
    }

    public override void Forward()
    {
        Context context = ConvolutionalNeuralNetwork.Context;
        Accelerator accelerator = ConvolutionalNeuralNetwork.Accelerator;

        var sumKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(MeanKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            _deviceInfos[i] = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });
            _deviceSums[i] = accelerator.Allocate1D<float>(3);

            Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

            for (int j = 0; j < _batchSize; j++)
            {
                _deviceInputs[i, j] = _inputs[i, j].Allocate(accelerator);

                sumKernal(index, _deviceInputs[i, j].View, _deviceSums[i].View, _deviceInfos[i].View);
            }
        }

        accelerator.Synchronize();

        var varianceKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(VarianceKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            _mean[i] = (Color)_deviceSums[i] / (Infos(i).Area * _batchSize);

            _deviceMeans[i] = accelerator.Allocate1D(new Color[] { _mean[i] });
            _deviceVariances[i] = accelerator.Allocate1D<float>(3);

            Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

            for (int j = 0; j < _batchSize; j++)
            {
                varianceKernal(index, _deviceInputs[i, j].View, _deviceMeans[i].View, _deviceVariances[i].View, _deviceInfos[i].View);
            }
        }

        accelerator.Synchronize();

        var normalizeKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<SingleLayerInfo>>(ForwardKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            _sigma[i] = Color.Pow((Color)_deviceVariances[i] / (Infos(i).Area * _batchSize) + new Color(ConvolutionalNeuralNetwork.ASYMPTOTEERRORFACTOR), 0.5f);

            Index2D index = new(Infos(i).Width, Infos(i).Length);
            _deviceValues[i] = accelerator.Allocate1D(new Color[] { _mean[i], _weight[i] / _sigma[i], _bias[i] });

            for (int j = 0; j < _batchSize; j++)
            {
                _deviceOutputs[i, j] = Normalized[i, j].AllocateEmpty(accelerator);

                normalizeKernal(index, _deviceInputs[i, j].View, _deviceOutputs[i, j].View, _deviceValues[i].View, _deviceInfos[i].View);
            }
        }

        accelerator.Synchronize();
        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                Normalized[i, j].CopyFromBuffer(_deviceOutputs[i, j]);
                _deviceOutputs[i, j].Dispose();
                _deviceInputs[i, j].Dispose();
            }
            _deviceInfos[i].Dispose();
            _deviceSums[i].Dispose();
            _deviceMeans[i].Dispose();
            _deviceVariances[i].Dispose();
            _deviceValues[i].Dispose();
        }
    }

    public override (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] inputs, FeatureMap[,] outGradients)
    {
        BaseStartup(inputs, outGradients);

        if (_weight == null)
        {
            _weight = new ColorVector(_inputDimensions);
            _bias = new ColorVector(_inputDimensions);
            for (int i = 0; i < _inputDimensions; i++)
            {
                _weight[i] = new Color(1);
                _bias[i] = new Color(0);
            }
        }

        _mean = new ColorVector(_inputDimensions);
        _sigma = new ColorVector(_inputDimensions);

        _deviceInfos = new MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense>[_inputDimensions];
        _deviceMeans = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions];
        _deviceGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions];
        _deviceValues = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions];
        _deviceSums = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions];
        _deviceVariances = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions];

        return (_outputs, _inGradients);
    }

    private static void BackwardsKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> inGradient, ArrayView<float> outGradient, ArrayView<Color> values, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        outGradient[mapsIndex * 3 + index.Z] = (inGradient[mapsIndex] * values[0] + values[1] * (input[mapsIndex] - values[2]) + values[3]).Clamp(1)[index.Z];
    }

    private static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> normalized, ArrayView<Color> values, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        normalized[mapsIndex] = (input[mapsIndex] - values[0]) * values[1] + values[2];
    }

    private static void GradientsKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> inGradient, ArrayView<Color> normalized, ArrayView<Color> mean, ArrayView<float> gradients, ArrayView<SingleLayerInfo> layer)
    {
        int gradientIndex = layer[0].Index(index.X, index.Y);
        float gradient = inGradient[gradientIndex][index.Z];
        Atomic.Add(ref gradients[index.Z], gradient * normalized[gradientIndex][index.Z]);
        Atomic.Add(ref gradients[index.Z + 3], gradient);
        Atomic.Add(ref gradients[index.Z + 6], gradient * (input[gradientIndex][index.Z] - mean[0][index.Z]));
    }

    private static void MeanKernal(Index3D index, ArrayView<Color> input, ArrayView<float> mean, ArrayView<SingleLayerInfo> info)
    {
        int inputIndex = info[0].Index(index.X, index.Y);
        Atomic.Add(ref mean[index.Z], input[inputIndex][index.Z]);
    }

    private static void VarianceKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> mean, ArrayView<float> variance, ArrayView<SingleLayerInfo> info)
    {
        int inputIndex = info[0].Index(index.X, index.Y);
        float difference = input[inputIndex][index.Z] - mean[0][index.Z];
        Atomic.Add(ref variance[index.Z], difference * difference);
    }

    private SingleLayerInfo Infos(int index)
    {
        return (SingleLayerInfo)_layerInfos[index];
    }

    public override void Reset()
    {
        for (int i = 0; i < _inputDimensions; i++)
        {
            _weight[i] = new Color(1);
            _bias[i] = new Color(0);
        }
    }

    public override void BackwardsNoUpdate()
    {
        Backwards(0);
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct Gradients
    {
        private Color _weightGradient;
        private Color _biasGradient;
        private Color _sigmaGradient;
        public Color WeightGradient => _weightGradient;
        public Color BiasGradient => _biasGradient;
        public Color SigmaGradient { get => _sigmaGradient; set => _sigmaGradient = value; }
        public Color MeanGradient { get; set; }
        public Color Sigma { get; set; }
        public Color Mean { get; set; }

        public void CopyFromBuffer(MemoryBuffer1D<float, Stride1D.Dense> floats)
        {
            unsafe
            {
                fixed (void* startAddress = &_weightGradient)
                    floats.AsArrayView<float>(0, 9).CopyToCPU(new Span<float>(startAddress, 9));
            }
        }
    }
}