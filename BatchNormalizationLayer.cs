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
    private readonly FeatureMap[][] _normalized;
    private readonly FeatureMap[][] _dL_dPNext;
    private readonly ColorVector _mean;
    private readonly ColorVector _sigma;
    private int _threadsWorking;
    public BatchNormalizationLayer(ref FeatureMap[][] input) : base(input.Length, 0,0)
    {
        _weight = new ColorVector(_dimensions);
        _bias = new ColorVector(_dimensions);
        for (int i = 0; i < _dimensions; i++)
        {
            _weight[i] = new Color(1, 1, 1);
            _bias[i] = new Color();
        }

        _mean = new ColorVector(_dimensions);
        _sigma = new ColorVector(_dimensions);

        _normalized = new FeatureMap[_dimensions][];
        _dL_dPNext = new FeatureMap[_dimensions][];
        for (int i = 0; i < _dimensions; i++)
        {
            _normalized[i] = new FeatureMap[input[i].Length];
            _dL_dPNext[i] = new FeatureMap[input[i].Length];
            for (int j = 0; j < input[i].Length; j++)
            {
                _dL_dPNext[i][j] = new FeatureMap(input[i][j].Width, input[i][j].Length);
                _normalized[i][j] = new FeatureMap(input[i][j].Width, input[i][j].Length);
            }
        }
        input = _normalized;
    }

    public override FeatureMap[][] Backwards(FeatureMap[][] input, FeatureMap[][] dL_dP, float learningRate)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);
        for (int i = 0; i < _dimensions; i++)
        {
            ThreadPool.QueueUserWorkItem(BackwardsThread, (i, input[i], dL_dP[i], _dL_dPNext[i], learningRate, accelerator));
        }

        do
            Thread.Sleep(100);
        while (_threadsWorking > 0);

        return _dL_dPNext;
    }

    public override FeatureMap[][] Forward(FeatureMap[][] input)
    {
        using Context context = Context.Create(builder => builder.Cuda());
        using Accelerator accelerator = context.CreateCudaAccelerator(0);
        for (int i = 0; i < _dimensions; i++)
        {
            ThreadPool.QueueUserWorkItem(ForwardThread, (i, input[i], _normalized[i], accelerator));
        }

        do
            Thread.Sleep(100);
        while (_threadsWorking > 0);

        return _normalized;
    }

    private void BackwardsThread(object? stateInfo)
    {
        if (stateInfo == null)
            throw new ArgumentNullException(nameof(stateInfo));
        (int dimension, FeatureMap[] input, FeatureMap[] dL_dP, FeatureMap[] dL_dPNext, float learningRate, Accelerator accelerator) = ((int, FeatureMap[], FeatureMap[], FeatureMap[], float, Accelerator))stateInfo;

        Interlocked.Increment(ref _threadsWorking);
        lock (dL_dPNext)
        {
            int batches = input.Length;
            int x = input[0].Width;
            int y = input[0].Length;
            float m = input[0].Area * batches;
            float _m = 1 / m;

            Color dL_dW = new();
            Color sum_dL_dP = new();
            Color dL_dS = new();
            for (int i = 0; i < batches; i++)
            {
                for (int j = 0; j < x; j++)
                {
                    for (int k = 0; k < y; k++)
                    {
                        dL_dP[i][j, k] *= _normalized[dimension][i][j, k].ReLUPropogation();
                        dL_dW += dL_dP[i][j, k] * _normalized[dimension][i][j, k];
                        sum_dL_dP += dL_dP[i][j, k];
                        dL_dS += dL_dP[i][j, k] * input[i][j, k];
                    }
                }
            }
            dL_dS -= _mean[dimension] * m;
            dL_dS *= Color.Pow(_sigma[dimension], -1.5f) * _weight[dimension] * -0.5f;
            Color dl_dM = -sum_dL_dP * _weight[dimension] / _sigma[dimension];

            using MemoryBuffer1D<int, Stride1D.Dense> deviceWidth = accelerator.Allocate1D(new int[] { x });
            using MemoryBuffer1D<Color, Stride1D.Dense> deviceValues = accelerator.Allocate1D(new Color[] { _sigma[dimension], 2 * _m * dL_dS, _mean[dimension], _m * dl_dM });
            MemoryBuffer1D<Color, Stride1D.Dense>[] deviceDL_dPNext = new MemoryBuffer1D<Color, Stride1D.Dense>[batches];
            Index2D index = new Index2D(x, y);
            for (int i = 0; i < batches; i++)
            {
                using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = input[i].Allocate(accelerator);
                using MemoryBuffer1D<Color, Stride1D.Dense> deviceDL_dP = dL_dP[i].Allocate(accelerator);
                deviceDL_dPNext[i] = _dL_dPNext[dimension][i].AllocateEmpty(accelerator);
                Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<int>> next =
                    accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<int>>(Next);

                next(index, deviceInput.View, deviceDL_dP.View, deviceDL_dPNext[i].View, deviceValues.View, deviceWidth.View);
            }

            accelerator.Synchronize();

            for (int i = 0; i < batches; i++)
            {
                _dL_dPNext[dimension][i].CopyFromBuffer(deviceDL_dPNext[i]);
                deviceDL_dPNext[i].Dispose();
            }

            _weight[dimension] -= learningRate * dL_dW.Clamp(1);
            _bias[dimension] -= learningRate * sum_dL_dP.Clamp(1);
        }
        Interlocked.Decrement(ref _threadsWorking);
    }

    private void ForwardThread(object? stateInfo)
    {
        if(stateInfo == null)
            throw new ArgumentNullException(nameof(stateInfo));
        (int dimension, FeatureMap[] input, FeatureMap[] normalized, Accelerator accelerator) = ((int, FeatureMap[], FeatureMap[], Accelerator))stateInfo;
        Interlocked.Increment(ref _threadsWorking);
        lock (normalized)
        {
            int batches = input.Length;
            int x = input[0].Width;
            int y = input[0].Length;
            float m = input[0].Area * batches;
            float _m = 1 / m;

            Color sum = new();
            for (int i = 0; i < batches; i++)
            {
                for (int j = 0; j < x; j++)
                {
                    for (int k = 0; k < y; k++)
                    {
                        sum += input[i][j, k];
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
                        sigma2 += Color.Pow(input[i][j, k] - _mean[dimension], 2);
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
                using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = input[i].Allocate(accelerator);
                deviceNormalized[i] = _normalized[dimension][i].AllocateEmpty(accelerator);

                Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<int>> normalizeKernal =
                    accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<int>>(Normalize);
                Index2D index = new Index2D(x, y);

                normalizeKernal(index, deviceInput.View, deviceNormalized[i].View, deviceValues.View, deviceWidth.View);
            }

            accelerator.Synchronize();

            for (int i = 0; i < batches; i++)
            {
                _normalized[dimension][i].CopyFromBuffer(deviceNormalized[i]);
                deviceNormalized[i].Dispose();
            }
        }
        Interlocked.Decrement(ref _threadsWorking);
    }

    private static void Next(Index2D index, ArrayView<Color> input, ArrayView<Color> dL_dP, ArrayView<Color> dL_dPNext, ArrayView<Color> values, ArrayView<int> width)
    {
        int position = index.Y * width[0] + index.X;
        dL_dPNext[position] = (dL_dP[position] / values[0] + values[1] * (input[position] - values[2]) + values[3]).Clamp(1);
    }

    private static void Normalize(Index2D index, ArrayView<Color> input, ArrayView<Color> normalized, ArrayView<Color> values, ArrayView<int> width)
    {
        int position = index.Y * width[0] + index.X;
        normalized[position] = ((input[position] - values[0]) / values[1]).ReLU() * values[2] + values[3];
    }
}

