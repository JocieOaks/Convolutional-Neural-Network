using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

public class NormalizationLayer : Layer
{
    [JsonProperty] private ColorVector _bias;
    [JsonProperty] private ColorVector _weight;
    private readonly FeatureMap[][] _normalized;
    private readonly FeatureMap[][] _dL_dPNext;
    private readonly ColorVector _mean;
    private readonly ColorVector _sigma;
    private int _threadsWorking;
    public NormalizationLayer(int dimensions,  ref FeatureMap[][] input) : base(dimensions, 0,0)
    {
        _weight = new ColorVector(dimensions);
        _bias = new ColorVector(dimensions);
        for (int i = 0; i < dimensions; i++)
        {
            _weight[i] = new Color(1, 1, 1);
            _bias[i] = new Color(0.5f, 0.5f, 0.5f);
        }

        _mean = new ColorVector(dimensions);
        _sigma = new ColorVector(dimensions);

        _normalized = new FeatureMap[dimensions][];
        _dL_dPNext = new FeatureMap[dimensions][];
        for (int i = 0; i < dimensions; i++)
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
        for(int i = 0; i < _dimensions; i++)
        {
            ThreadPool.QueueUserWorkItem(BackwardsThread, (i, input[i], dL_dP[i], _dL_dPNext[i], learningRate));
        }

        do
            Thread.Sleep(100);
        while (_threadsWorking > 0);

        return _dL_dPNext;
    }

    public override FeatureMap[][] Forward(FeatureMap[][] input)
    {
        for(int i = 0; i < _dimensions; i++)
        {
            ThreadPool.QueueUserWorkItem(ForwardThread, (i, input[i], _normalized[i]));
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
        (int dimension, FeatureMap[] input, FeatureMap[] dL_dP, FeatureMap[] dL_dPNext, float learningRate) = ((int, FeatureMap[], FeatureMap[], FeatureMap[], float))stateInfo;

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

            for (int i = 0; i < batches; i++)
            {
                for (int j = 0; j < x; j++)
                {
                    for (int k = 0; k < y; k++)
                    {
                        dL_dPNext[i][j, k] = (dL_dP[i][j, k] / _sigma[dimension] + 2 * _m * dL_dS * (input[i][j, k] - _mean[dimension]) + _m * dl_dM).Clamp(1);
                    }
                }
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
        (int dimension, FeatureMap[] input, FeatureMap[] normalized) = ((int, FeatureMap[], FeatureMap[]))stateInfo;
        Interlocked.Increment(ref _threadsWorking);
        lock (normalized)
        {
            int batches = input.Length;
            int x = input[0].Width;
            int y = input[0].Length;
            float m = input[0].Area * batches;
            float _m = 1 / m;

            _mean[dimension] = new();
            for (int i = 0; i < batches; i++)
            {
                for (int j = 0; j < x; j++)
                {
                    for (int k = 0; k < y; k++)
                    {
                        _mean[dimension] += input[i][j, k];
                    }
                }
            }
            _mean[dimension] = _m * _mean[dimension];

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

            for (int i = 0; i < batches; i++)
            {
                for (int j = 0; j < x; j++)
                {
                    for (int k = 0; k < y; k++)
                    {
                        normalized[i][j, k] = ((input[i][j, k] - _mean[dimension]) / _sigma[dimension]).ReLU() * _weight[dimension] + _bias[dimension];
                    }
                }
            }
        }
        Interlocked.Decrement(ref _threadsWorking);
    }
}

