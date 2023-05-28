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

public class BatchNormalizationLayer : Layer
{
    [JsonProperty] private ColorVector _bias;
    [JsonProperty] private ColorVector _weight;
    private FeatureMap[,] Normalized => _outputs;
    private readonly ColorVector _mean;
    private readonly ColorVector _sigma;
    private int _threadsWorking;
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

        int batches = input.GetLength(1);
        int x = input[dimension, 0].Width;
        int y = input[dimension, 0].Length;
        float m = input[dimension, 0].Area * batches;
        float _m = 1 / m;

        Color weightGradient = new();
        Color biasGradient = new();
        Color sigmaGradient = new();
        for (int i = 0; i < batches; i++)
        {
            for (int j = 0; j < x; j++)
            {
                for (int k = 0; k < y; k++)
                {
                    inGradient[dimension, i][j, k] *= Normalized[dimension, i][j, k].ReLUPropogation();
                    weightGradient += inGradient[dimension, i][j, k] * Normalized[dimension, i][j, k];
                    biasGradient += inGradient[dimension, i][j, k];
                    sigmaGradient += inGradient[dimension, i][j, k] * input[dimension,i][j, k];
                }
            }
        }
        sigmaGradient -= _mean[dimension] * m;
        sigmaGradient *= Color.Pow(_sigma[dimension], -1.5f) * _weight[dimension] * -0.5f;
        Color meanGradient = -biasGradient * _weight[dimension] / _sigma[dimension];

        using MemoryBuffer1D<int, Stride1D.Dense> deviceWidth = accelerator.Allocate1D(new int[] { x });
        using MemoryBuffer1D<Color, Stride1D.Dense> deviceValues = accelerator.Allocate1D(new Color[] { _sigma[dimension], 2 * _m * sigmaGradient, _mean[dimension], _m * meanGradient });
        MemoryBuffer1D<Color, Stride1D.Dense>[] deviceOutGradient = new MemoryBuffer1D<Color, Stride1D.Dense>[batches];
        Index2D index = new Index2D(x, y);
        for (int i = 0; i < batches; i++)
        {
            using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = input[dimension, i].Allocate(accelerator);
            using MemoryBuffer1D<Color, Stride1D.Dense> deviceInGradient = inGradient[dimension, i].Allocate(accelerator);
            deviceOutGradient[i] = _outGradients[dimension,i].AllocateEmpty(accelerator);
            Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<int>> next =
                accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<int>>(Next);

            next(index, deviceInput.View, deviceInGradient.View, deviceOutGradient[i].View, deviceValues.View, deviceWidth.View);
        }

        accelerator.Synchronize();

        for (int i = 0; i < batches; i++)
        {
            _outGradients[dimension, i].CopyFromBuffer(deviceOutGradient[i]);
            deviceOutGradient[i].Dispose();
        }

        _weight[dimension] -= learningRate * weightGradient.Clamp(1);
        _bias[dimension] -= learningRate * biasGradient.Clamp(1);
        
        Interlocked.Decrement(ref _threadsWorking);
    }

    private void ForwardThread(object? stateInfo)
    {
        if (stateInfo == null)
            throw new ArgumentNullException(nameof(stateInfo));
        (int dimension, FeatureMap[,] input, Accelerator accelerator) = ((int, FeatureMap[,], Accelerator))stateInfo;
        Interlocked.Increment(ref _threadsWorking);

        int batches = input.GetLength(1);
        int x = input[dimension, 0].Width;
        int y = input[dimension, 0].Length;
        float m = input[dimension, 0].Area * batches;
        float _m = 1 / m;

        Color sum = new();
        for (int i = 0; i < batches; i++)
        {
            for (int j = 0; j < x; j++)
            {
                for (int k = 0; k < y; k++)
                {
                    sum += input[dimension, i][j, k];
                }
            }
        }
        _mean[dimension] = _m * sum;

        Color sigma2 = new();

        for (int i = 0; i < batches; i++)
        {
            for (int j = 0; j < x; j++)
            {
                for (int k = 0; k < y; k++)
                {
                    sigma2 += Color.Pow(input[dimension, i][j, k] - _mean[dimension], 2);
                }
            }
        }

        sigma2 = _m * sigma2;
        _sigma[dimension] = Color.Pow(sigma2 + new Color(CLIP.ASYMPTOTEERRORFACTOR, CLIP.ASYMPTOTEERRORFACTOR, CLIP.ASYMPTOTEERRORFACTOR), 0.5f);

        using MemoryBuffer1D<int, Stride1D.Dense> deviceWidth = accelerator.Allocate1D(new int[] { x });
        using MemoryBuffer1D<Color, Stride1D.Dense> deviceValues = accelerator.Allocate1D(new Color[] { _mean[dimension], _sigma[dimension], _weight[dimension], _bias[dimension] });
        MemoryBuffer1D<Color, Stride1D.Dense>[] deviceNormalized = new MemoryBuffer1D<Color, Stride1D.Dense>[batches];
        for (int i = 0; i < batches; i++)
        {
            using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = input[dimension, i].Allocate(accelerator);
            deviceNormalized[i] = Normalized[dimension, i].AllocateEmpty(accelerator);

            Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<int>> normalizeKernal =
                accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<int>>(Normalize);
            Index2D index = new Index2D(x, y);

            normalizeKernal(index, deviceInput.View, deviceNormalized[i].View, deviceValues.View, deviceWidth.View);
        }

        accelerator.Synchronize();

        for (int i = 0; i < batches; i++)
        {
            Normalized[dimension, i].CopyFromBuffer(deviceNormalized[i]);
            deviceNormalized[i].Dispose();
        }

        Interlocked.Decrement(ref _threadsWorking);
    }

    private static void Next(Index2D index, ArrayView<Color> input, ArrayView<Color> inGradient, ArrayView<Color> outGradient, ArrayView<Color> values, ArrayView<int> width)
    {
        int position = index.Y * width[0] + index.X;
        outGradient[position] = (inGradient[position] / values[0] + values[1] * (input[position] - values[2]) + values[3]).Clamp(1);
    }

    private static void Normalize(Index2D index, ArrayView<Color> input, ArrayView<Color> normalized, ArrayView<Color> values, ArrayView<int> width)
    {
        int position = index.Y * width[0] + index.X;
        normalized[position] = ((input[position] - values[0]) / values[1]).ReLU() * values[2] + values[3];
    }
}

