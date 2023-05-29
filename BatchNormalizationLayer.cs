using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using System.Runtime.InteropServices;
using System.Security.AccessControl;

public class BatchNormalizationLayer : Layer
{
    [JsonProperty] private ColorVector _bias;
    [JsonProperty] private ColorVector _weight;
    private FeatureMap[,] Normalized => _outputs;
    private readonly ColorVector _mean;
    private readonly ColorVector _sigma;
    private int _threadsWorking;

    public override string Name => "Batch Normalization Layer";
    public BatchNormalizationLayer(ref FeatureMap[,] input) : base(1,1, ref input)
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

    public override FeatureMap[,] Backwards(FeatureMap[,] input, FeatureMap[,] inGradients, float learningRate)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);
        for (int i = 0; i < _inputDimensions; i++)
        {
            ThreadPool.QueueUserWorkItem(BackwardsThread, (i, input, inGradients, learningRate, accelerator));
        }

        do
            Thread.Sleep(100);
        while (_threadsWorking > 0);

        return _outGradients;
    }

    public override FeatureMap[,] Forward(FeatureMap[,] input)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);
        for (int i = 0; i < _inputDimensions; i++)
        {
            ThreadPool.QueueUserWorkItem(ForwardThread, (i, input, accelerator));
        }

        do
            Thread.Sleep(100);
        while (_threadsWorking > 0);

        return Normalized;
    }

    private void BackwardsThread(object? stateInfo)
    {
        if (stateInfo == null)
            throw new ArgumentNullException(nameof(stateInfo));
        (int dimension, FeatureMap[,] input, FeatureMap[,] inGradient, float learningRate, Accelerator accelerator) = ((int, FeatureMap[,], FeatureMap[,], float, Accelerator))stateInfo;

        Interlocked.Increment(ref _threadsWorking);

        SingleLayerInfo info = Infos(dimension);
        float m = info.Area * _batchSize;
        float _m = 1 / m;

        AcceleratorStream stream = accelerator.CreateStream();

        MemoryBuffer1D<Color, Stride1D.Dense>[] deviceInputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_batchSize];
        MemoryBuffer1D<Color, Stride1D.Dense>[] deviceInGradients = new MemoryBuffer1D<Color, Stride1D.Dense>[_batchSize];
        using MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense> deviceInfo = accelerator.Allocate1D(new SingleLayerInfo[] { info });
        using MemoryBuffer1D<float, Stride1D.Dense> deviceGradients = accelerator.Allocate1D<float>(9);
        Index3D index = new(info.Width, info.Length, 3);

        for (int i = 0; i < _batchSize; i++)
        {
            deviceInputs[i] = input[dimension, i].Allocate(accelerator);
            deviceInGradients[i] = inGradient[dimension, i].Allocate(accelerator);
            using MemoryBuffer1D<Color, Stride1D.Dense> deviceNormalized = Normalized[dimension, i].Allocate(accelerator);
            Action<AcceleratorStream, Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>> gradientKernal =
                accelerator.LoadAutoGroupedKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(GradientsKernal);

            gradientKernal(stream, index, deviceInputs[i].View, deviceInGradients[i].View, deviceNormalized.View, deviceGradients.View, deviceInfo.View);
        }

        stream.Synchronize();

        Gradients gradients = new();
        gradients.CopyFromBuffer(deviceGradients);

        gradients.SigmaGradient -= _mean[dimension] * m;
        gradients.SigmaGradient *= Color.Pow(_sigma[dimension], -1.5f) * _weight[dimension] * -0.5f;
        gradients.MeanGradient = -gradients.BiasGradient * _weight[dimension] / _sigma[dimension];

        using MemoryBuffer1D<Color, Stride1D.Dense> deviceValues = accelerator.Allocate1D(new Color[] { _sigma[dimension], 2 * _m * gradients.SigmaGradient, _mean[dimension], _m * gradients.MeanGradient });
        MemoryBuffer1D<float, Stride1D.Dense>[] deviceOutGradient = new MemoryBuffer1D<float, Stride1D.Dense>[_batchSize];
        for (int i = 0; i < _batchSize; i++)
        {
            deviceOutGradient[i] = _outGradients[dimension,i].AllocateFloat(accelerator);
            Action<AcceleratorStream, Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<Color>, ArrayView<SingleLayerInfo>> next =
                accelerator.LoadAutoGroupedKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<Color>, ArrayView<SingleLayerInfo>>(BackwardsKernal);

            next(stream, index, deviceInputs[i].View, deviceInGradients[i].View, deviceOutGradient[i].View, deviceValues.View, deviceInfo.View);

            deviceInputs[i].Dispose();
            deviceInGradients[i].Dispose();
        }

        stream.Synchronize();

        for (int i = 0; i < _batchSize; i++)
        {
            _outGradients[dimension, i].CopyFromBuffer(deviceOutGradient[i]);
            deviceOutGradient[i].Dispose();
        }

        _weight[dimension] -= learningRate * gradients.WeightGradient.Clamp(1);
        _bias[dimension] -= learningRate * gradients.BiasGradient.Clamp(1);
        
        Interlocked.Decrement(ref _threadsWorking);
    }

    private static void GradientsKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> inGradient, ArrayView<Color> normalized, ArrayView<float> gradients, ArrayView<SingleLayerInfo> layer)
    {
        int gradientIndex = layer[0].Index(index.X, index.Y);
        inGradient[gradientIndex] *= normalized[gradientIndex].ReLUPropogation();
        float gradient = inGradient[gradientIndex][index.Z];
        Atomic.Add(ref gradients[index.Z], gradient * normalized[gradientIndex][index.Z]);
        Atomic.Add(ref gradients[index.Z + 3], gradient);
        Atomic.Add(ref gradients[index.Z + 6], gradient * input[gradientIndex][index.Z]);
    }

    private (MemoryBuffer1D<Color, Stride1D.Dense>, MemoryBuffer1D<Color, Stride1D.Dense>) InitializeGradientsKernal(Accelerator accelerator, Index3D index, AcceleratorStream stream, FeatureMap input, FeatureMap inGradient, FeatureMap normalized, MemoryBuffer1D<float, Stride1D.Dense> deviceGradients, MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense> deviceInfo)
    {
        MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = input.Allocate(accelerator);
        MemoryBuffer1D<Color, Stride1D.Dense> deviceInGradient = inGradient.Allocate(accelerator);
        using MemoryBuffer1D<Color,Stride1D.Dense> deviceNormalized = normalized.Allocate(accelerator);
        Action<AcceleratorStream, Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>> gradientKernal =
            accelerator.LoadAutoGroupedKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(GradientsKernal);

        gradientKernal(stream, index, deviceInput.View, deviceInGradient.View, deviceNormalized.View, deviceGradients.View, deviceInfo.View);

        return (deviceInput, deviceInGradient);
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

    private void ForwardThread(object? stateInfo)
    {
        if (stateInfo == null)
            throw new ArgumentNullException(nameof(stateInfo));
        (int dimension, FeatureMap[,] input, Accelerator accelerator) = ((int, FeatureMap[,], Accelerator))stateInfo;
        Interlocked.Increment(ref _threadsWorking);

        AcceleratorStream stream = accelerator.CreateStream();

        SingleLayerInfo info = Infos(dimension);
        float m = info.Area * _batchSize;
        float _m = 1 / m;

        Color sum = new();
        for (int i = 0; i < _batchSize; i++)
        {
            for (int j = 0; j < info.Width; j++)
            {
                for (int k = 0; k < info.Length; k++)
                {
                    sum += input[dimension, i][j, k];
                }
            }
        }
        _mean[dimension] = _m * sum;

        Color sigma2 = new();

        for (int i = 0; i < _batchSize; i++)
        {
            for (int j = 0; j < info.Width; j++)
            {
                for (int k = 0; k < info.Length; k++)
                {
                    sigma2 += Color.Pow(input[dimension, i][j, k] - _mean[dimension], 2);
                }
            }
        }

        sigma2 = _m * sigma2;
        _sigma[dimension] = Color.Pow(sigma2 + new Color(CLIP.ASYMPTOTEERRORFACTOR, CLIP.ASYMPTOTEERRORFACTOR, CLIP.ASYMPTOTEERRORFACTOR), 0.5f);

        using MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense> deviceInfo = accelerator.Allocate1D(new SingleLayerInfo[] { info });
        using MemoryBuffer1D<Color, Stride1D.Dense> deviceValues = accelerator.Allocate1D(new Color[] { _mean[dimension], _sigma[dimension], _weight[dimension], _bias[dimension] });
        MemoryBuffer1D<Color, Stride1D.Dense>[] deviceNormalized = new MemoryBuffer1D<Color, Stride1D.Dense>[_batchSize];
        Index2D index = new(info.Width, info.Length);
        for (int i = 0; i < _batchSize; i++)
        {
            using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = input[dimension, i].Allocate(accelerator);
            deviceNormalized[i] = Normalized[dimension, i].AllocateEmpty(accelerator);

            Action<AcceleratorStream, Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<SingleLayerInfo>> normalizeKernal =
                accelerator.LoadAutoGroupedKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<SingleLayerInfo>>(ForwardKernal);
            

            normalizeKernal(stream, index, deviceInput.View, deviceNormalized[i].View, deviceValues.View, deviceInfo.View);
        }

        accelerator.Synchronize();

        for (int i = 0; i < _batchSize; i++)
        {
            Normalized[dimension, i].CopyFromBuffer(deviceNormalized[i]);
            deviceNormalized[i].Dispose();
        }

        Interlocked.Decrement(ref _threadsWorking);
    }

    private SingleLayerInfo Infos(int index)
    {
        return (SingleLayerInfo)_layerInfos[index];
    }

    private static void BackwardsKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> inGradient, ArrayView<float> outGradient, ArrayView<Color> values, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        outGradient[mapsIndex * 3 + index.Z] = (inGradient[mapsIndex] / values[0] + values[1] * (input[mapsIndex] - values[2]) + values[3]).Clamp(1)[index.Z];
    }

    private static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> normalized, ArrayView<Color> values, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        normalized[mapsIndex] = ((input[mapsIndex] - values[0]) / values[1]).ReLU() * values[2] + values[3];
    }
}

