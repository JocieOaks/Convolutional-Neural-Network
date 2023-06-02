using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;
using System.Runtime.InteropServices;

public class BatchNormalizationLayer : Layer
{
    private readonly ColorVector _mean;
    private readonly ColorVector _sigma;
    [JsonProperty] private readonly ColorVector _bias;
    [JsonProperty] private readonly ColorVector _weight;

    public BatchNormalizationLayer(ref FeatureMap[,] input) : base(1, 1, ref input)
    {
        _weight = new ColorVector(_inputDimensions);
        _bias = new ColorVector(_inputDimensions);
        for (int i = 0; i < _inputDimensions; i++)
        {
            _weight[i] = new Color(1, 1, 1);
            _bias[i] = new Color();
        }

        _mean = new ColorVector(_inputDimensions);
        _sigma = new ColorVector(_inputDimensions);
    }

    public override string Name => "Batch Normalization Layer";

    private FeatureMap[,] Normalized => _outputs;

    public override FeatureMap[,] Backwards(FeatureMap[,] inputs, FeatureMap[,] inGradients, float learningRate)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

        MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceInputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];
        MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceInGradients = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];
        MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense>[] deviceInfos = new MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense>[_inputDimensions];

        MemoryBuffer1D<float, Stride1D.Dense>[] deviceGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions];
        MemoryBuffer1D<float, Stride1D.Dense>[,] deviceOutGradient = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions, _batchSize];

        Gradients[] gradients = new Gradients[_inputDimensions];

        var gradientKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(GradientsKernal);
        
        for (int i = 0; i < _inputDimensions; i++)
        {
            deviceInfos[i] = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });
            deviceGradients[i] = accelerator.Allocate1D<float>(9);
            using var deviceMean = accelerator.Allocate1D(new Color[] { _mean[i] });
            Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

            for (int j = 0; j < _batchSize; j++)
            {
                deviceInputs[i, j] = inputs[i, j].Allocate(accelerator);
                deviceInGradients[i, j] = inGradients[i, j].Allocate(accelerator);
                using var deviceNormalized = Normalized[i, j].Allocate(accelerator);

                gradientKernal(index, deviceInputs[i, j].View, deviceInGradients[i, j].View, deviceNormalized.View, deviceMean.View, deviceGradients[i].View, deviceInfos[i].View);
            }
        }

        accelerator.Synchronize();

        var backwardsKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<Color>, ArrayView<SingleLayerInfo>>(BackwardsKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            gradients[i] = new();
            gradients[i].CopyFromBuffer(deviceGradients[i]);
            deviceGradients[i].Dispose();

            gradients[i].SigmaGradient *= Color.Pow(_sigma[i], -1.5f) * _weight[i] * -0.5f;
            gradients[i].MeanGradient = -gradients[i].BiasGradient * _weight[i] / _sigma[i];

            float invM = 1f / (Infos(i).Area * _batchSize);

            using var deviceValues = accelerator.Allocate1D(new Color[] { _weight[i] / _sigma[i], 2 * invM * gradients[i].SigmaGradient, _mean[i], invM * gradients[i].MeanGradient });

            Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

            for (int j = 0; j < _batchSize; j++)
            {
                deviceOutGradient[i, j] = _outGradients[i, j].AllocateFloat(accelerator);

                backwardsKernal(index, deviceInputs[i, j].View, deviceInGradients[i, j].View, deviceOutGradient[i, j].View, deviceValues.View, deviceInfos[i].View);
            }
        }

        accelerator.Synchronize();

        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _outGradients[i, j].CopyFromBuffer(deviceOutGradient[i, j]);
                deviceOutGradient[i, j].Dispose();
                deviceInputs[i, j].Dispose();
                deviceInGradients[i, j].Dispose();
            }

            _weight[i] -= learningRate * gradients[i].WeightGradient.Clamp(1);
            _bias[i] -= learningRate * gradients[i].BiasGradient.Clamp(1);

            deviceInfos[i].Dispose();
        }

        return _outGradients;
    }

    public override FeatureMap[,] Forward(FeatureMap[,] input)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

        MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceInputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];
        MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense>[] deviceInfos = new MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense>[_inputDimensions];

        MemoryBuffer1D<float, Stride1D.Dense>[] deviceSums = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions];
        MemoryBuffer1D<Color, Stride1D.Dense>[] deviceMeans = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions];
        MemoryBuffer1D<float, Stride1D.Dense>[] deviceVariances = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions];
        MemoryBuffer1D<Color, Stride1D.Dense>[] deviceValues = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions];
        MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceNormalized = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];

        var sumKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(MeanKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            deviceInfos[i] = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });
            deviceSums[i] = accelerator.Allocate1D<float>(3);

            Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

            for (int j = 0; j < _batchSize; j++)
            {
                deviceInputs[i, j] = input[i, j].Allocate(accelerator);

                sumKernal(index, deviceInputs[i, j].View, deviceSums[i].View, deviceInfos[i].View);
            }
        }

        accelerator.Synchronize();

        var varianceKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(VarianceKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            _mean[i] = (Color)deviceSums[i] / (Infos(i).Area * _batchSize);

            deviceMeans[i] = accelerator.Allocate1D(new Color[] { _mean[i] });
            deviceVariances[i] = accelerator.Allocate1D<float>(3);

            Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

            for (int j = 0; j < _batchSize; j++)
            {
                varianceKernal(index, deviceInputs[i, j].View, deviceMeans[i].View, deviceVariances[i].View, deviceInfos[i].View);
            }
        }

        accelerator.Synchronize();

        var normalizeKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<SingleLayerInfo>>(ForwardKernal);

        for (int i = 0; i < _inputDimensions; i++)
        {
            _sigma[i] = Color.Pow((Color)deviceVariances[i] / (Infos(i).Area * _batchSize) + new Color(CLIP.ASYMPTOTEERRORFACTOR, CLIP.ASYMPTOTEERRORFACTOR, CLIP.ASYMPTOTEERRORFACTOR), 0.5f);

            Index2D index = new(Infos(i).Width, Infos(i).Length);
            deviceValues[i] = accelerator.Allocate1D(new Color[] { _mean[i], _weight[i] / _sigma[i], _bias[i] });

            for (int j = 0; j < _batchSize; j++)
            {
                deviceNormalized[i, j] = Normalized[i, j].AllocateEmpty(accelerator);

                normalizeKernal(index, deviceInputs[i, j].View, deviceNormalized[i, j].View, deviceValues[i].View, deviceInfos[i].View);
            }
        }

        accelerator.Synchronize();
        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                Normalized[i, j].CopyFromBuffer(deviceNormalized[i, j]);
                deviceNormalized[i, j].Dispose();
                deviceInputs[i, j].Dispose();
            }
            deviceInfos[i].Dispose();
            deviceSums[i].Dispose();
            deviceMeans[i].Dispose();
            deviceVariances[i].Dispose();
            deviceValues[i].Dispose();
        }

        return Normalized;
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