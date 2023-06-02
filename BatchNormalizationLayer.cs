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
    private readonly ColorVector _mean;
    private readonly ColorVector _sigma;
    [JsonProperty] private ColorVector _bias;
    private int _threadsWorking;
    [JsonProperty] private ColorVector _weight;
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

    private static void BackwardsKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> inGradient, ArrayView<float> outGradient, ArrayView<Color> values, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        outGradient[mapsIndex * 3 + index.Z] = (inGradient[mapsIndex] * values[0] + values[1] * (input[mapsIndex] - values[2]) + values[3]).Clamp(1)[index.Z];
    }

    private static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> normalized, ArrayView<Color> values, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        normalized[mapsIndex] = ((input[mapsIndex] - values[0]) / values[1]) * values[2] + values[3];
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

    private static void SigmaKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> mean, ArrayView<float> sigma, ArrayView<SingleLayerInfo> info)
    {
        int inputIndex = info[0].Index(index.X, index.Y);
        float difference = input[inputIndex][index.Z] - mean[0][index.Z];
        Atomic.Add(ref sigma[index.Z], difference * difference);
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

        using AcceleratorStream stream = accelerator.CreateStream();

        MemoryBuffer1D<Color, Stride1D.Dense>[] deviceInputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_batchSize];
        MemoryBuffer1D<Color, Stride1D.Dense>[] deviceInGradients = new MemoryBuffer1D<Color, Stride1D.Dense>[_batchSize];
        using MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense> deviceInfo = accelerator.Allocate1D(new SingleLayerInfo[] { info });
        using MemoryBuffer1D<float, Stride1D.Dense> deviceGradients = accelerator.Allocate1D<float>(9);
        using MemoryBuffer1D<Color, Stride1D.Dense> deviceMean = accelerator.Allocate1D(new Color[] { _mean[dimension] });
        Index3D index = new(info.Width, info.Length, 3);

        for (int i = 0; i < _batchSize; i++)
        {
            deviceInputs[i] = input[dimension, i].Allocate(accelerator);
            deviceInGradients[i] = inGradient[dimension, i].Allocate(accelerator);
            using MemoryBuffer1D<Color, Stride1D.Dense> deviceNormalized = Normalized[dimension, i].Allocate(accelerator);
            Action<AcceleratorStream, Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>> gradientKernal =
                accelerator.LoadAutoGroupedKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(GradientsKernal);

            gradientKernal(stream, index, deviceInputs[i].View, deviceInGradients[i].View, deviceNormalized.View, deviceMean.View, deviceGradients.View, deviceInfo.View);
        }

        stream.Synchronize();

        Gradients gradients = new();
        gradients.CopyFromBuffer(deviceGradients);

        gradients.SigmaGradient *= Color.Pow(_sigma[dimension], -1.5f) * _weight[dimension] * -0.5f;
        gradients.MeanGradient = -gradients.BiasGradient * _weight[dimension] / _sigma[dimension];

        using MemoryBuffer1D<Color, Stride1D.Dense> deviceValues = accelerator.Allocate1D(new Color[] {_weight[dimension] / _sigma[dimension], 2 * _m * gradients.SigmaGradient, _mean[dimension], _m * gradients.MeanGradient });
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

    private void ForwardThread(object? stateInfo)
    {
        if (stateInfo == null)
            throw new ArgumentNullException(nameof(stateInfo));
        (int dimension, FeatureMap[,] input, Accelerator accelerator) = ((int, FeatureMap[,], Accelerator))stateInfo;
        Interlocked.Increment(ref _threadsWorking);

        using AcceleratorStream stream = accelerator.CreateStream();

        SingleLayerInfo info = Infos(dimension);
        float m = info.Area * _batchSize;
        float invM = 1 / m;

        using MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense> deviceInfo = accelerator.Allocate1D(new SingleLayerInfo[] { info });
        using MemoryBuffer1D<float, Stride1D.Dense> deviceSum = accelerator.Allocate1D<float>(3);
        MemoryBuffer1D<Color, Stride1D.Dense>[] deviceInputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_batchSize];
        Index3D index3 = new(info.Width, info.Length, 3);

        for (int i = 0; i < _batchSize; i++)
        {
            deviceInputs[i] = input[dimension, i].Allocate(accelerator);
            Action<AcceleratorStream, Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>> sumKernal =
                        accelerator.LoadAutoGroupedKernel<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(MeanKernal);

            sumKernal(stream, index3, deviceInputs[i].View, deviceSum.View, deviceInfo.View);
        }

        stream.Synchronize();

        _mean[dimension] = invM * (Color)deviceSum;

        using MemoryBuffer1D<Color, Stride1D.Dense> deviceMean = accelerator.Allocate1D(new Color[] { _mean[dimension] });
        using MemoryBuffer1D<float, Stride1D.Dense> deviceSigma2 = accelerator.Allocate1D<float>(3);

        for (int i = 0; i < _batchSize; i++)
        {
            Action<AcceleratorStream, Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>> sigmaKernal =
                accelerator.LoadAutoGroupedKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(SigmaKernal);

            sigmaKernal(stream, index3, deviceInputs[i].View, deviceMean.View, deviceSigma2.View, deviceInfo.View);
        }

        stream.Synchronize();

        _sigma[dimension] = Color.Pow(invM * (Color)deviceSigma2 + new Color(CLIP.ASYMPTOTEERRORFACTOR, CLIP.ASYMPTOTEERRORFACTOR, CLIP.ASYMPTOTEERRORFACTOR), 0.5f);

        Index2D index2 = new(info.Width, info.Length);
        using MemoryBuffer1D<Color, Stride1D.Dense> deviceValues = accelerator.Allocate1D(new Color[] { _mean[dimension], _sigma[dimension], _weight[dimension], _bias[dimension] });
        MemoryBuffer1D<Color, Stride1D.Dense>[] deviceNormalized = new MemoryBuffer1D<Color, Stride1D.Dense>[_batchSize];
        

        for (int i = 0; i < _batchSize; i++)
        {
            deviceNormalized[i] = Normalized[dimension, i].AllocateEmpty(accelerator);

            Action<AcceleratorStream, Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<SingleLayerInfo>> normalizeKernal =
                accelerator.LoadAutoGroupedKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<SingleLayerInfo>>(ForwardKernal);

            normalizeKernal(stream, index2, deviceInputs[i].View, deviceNormalized[i].View, deviceValues.View, deviceInfo.View);
        }

        stream.Synchronize();

        for (int i = 0; i < _batchSize; i++)
        {
            Normalized[dimension, i].CopyFromBuffer(deviceNormalized[i]);
            deviceNormalized[i].Dispose();
            deviceInputs[i].Dispose();
        }

        Interlocked.Decrement(ref _threadsWorking);
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

