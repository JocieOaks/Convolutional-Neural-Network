using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

public class NormalizationLayer : Layer
{
    ColorVector _bias;
    ColorVector _weight;
    readonly FeatureMap[][] normalized;
    readonly ColorVector mean;
    readonly ColorVector sigma;
    public NormalizationLayer(int dimensions) : base(dimensions, 0,0)
    {
        _weight = new ColorVector(dimensions);
        _bias = new ColorVector(dimensions);
        for (int i = 0; i < dimensions; i++)
        {
            _weight[i] = new Color(1, 1, 1);
            _bias[i] = new Color(0.5f, 0.5f, 0.5f);
        }

        mean = new ColorVector(dimensions);
        sigma = new ColorVector(dimensions);
        normalized = new FeatureMap[dimensions][];
    }

    public override FeatureMap[][] Backwards(FeatureMap[][] dL_dP, FeatureMap[][] input, float alpha)
    {
        FeatureMap[][] dL_dPNext = new FeatureMap[_dimensions][];

        for(int i = 0; i < _dimensions; i++)
        {
            dL_dPNext[i] = Backwards(dL_dP[i], input[i], i, alpha);
        }

        return dL_dPNext;
    }

    public override FeatureMap[][] Forward(FeatureMap[][] input)
    {
        for(int i = 0; i < _dimensions; i++)
        {
            normalized[i] = Forward(input[i], i);
        }

        return normalized;
    }
    
    private FeatureMap[] Backwards(FeatureMap[] dL_dP, FeatureMap[] input, int dimension, float alpha)
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
                    dL_dP[i][j, k] *= normalized[dimension][i][j, k].ReLUPropogation();
                    dL_dW += dL_dP[i][j, k] * normalized[dimension][i][j, k];
                    sum_dL_dP += dL_dP[i][j, k];
                    dL_dS += dL_dP[i][j, k] * input[i][j, k];
                }
            }
        }
        dL_dS -= mean[dimension] * m;
        dL_dS *= Color.Pow(sigma[dimension], -1.5f) * _weight[dimension] * -0.5f;
        Color dl_dM = -sum_dL_dP * _weight[dimension] / sigma[dimension];

        FeatureMap[] dL_dPNext = new FeatureMap[batches];

        for (int i = 0; i < batches; i++)
        {
            dL_dPNext[i] = new FeatureMap(x, y);
            for (int j = 0; j < x; j++)
            {
                for (int k = 0; k < y; k++)
                {
                    dL_dPNext[i][j, k] = (dL_dP[i][j, k] / sigma[dimension] + 2 * _m  * dL_dS * (input[i][j, k] - mean[dimension]) + _m * dl_dM).Clamp(1);
                }
            }
        }

        _weight[dimension] -= alpha * dL_dW.Clamp(1);
        _bias[dimension] -= alpha * sum_dL_dP.Clamp(1);

        return dL_dPNext;
    }

    private FeatureMap[] Forward(FeatureMap[] input, int dimension)
    {
        int batches = input.Length;
        int x = input[0].Width;
        int y = input[0].Length;
        float m = input[0].Area * batches;
        float _m = 1 / m;

        mean[dimension] = new();
        for(int i = 0; i < batches; i++)
        {
            for (int j = 0; j < x; j++)
            {
                for (int k = 0; k < y; k++)
                {
                    mean[dimension] += input[i][j, k];
                }
            }
        }
        mean[dimension] = _m * mean[dimension];

        Color sigma2 = new();

        for(int i = 0; i < batches; i++)
        {
            for (int j = 0; j < x; j++)
            {
                for (int k = 0; k < y; k++)
                {
                    sigma2 += Color.Pow(input[i][j, k] - mean[dimension], 2);
                }
            }
        }

        sigma2 = _m * sigma2;
        sigma[dimension] = Color.Pow(sigma2 + new Color(0.001f, 0.001f, 0.001f), 0.5f);

        FeatureMap[] output = new FeatureMap[batches];
        for(int i = 0; i < batches; i++)
        {
            output[i] = new FeatureMap(x, y);
            for (int j = 0; j < x; j++)
            {
                for (int k = 0; k < y; k++)
                {
                    output[i][j, k] =((input[i][j, k] - mean[dimension]) / sigma[dimension]).ReLU() * _weight[dimension] + _bias[dimension];
                }
            }
        }

        return output;
    }
}

