using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

public class NormalizationLayer<T> : Layer<T> where T : IDot<T>, new()
{
    T[] _weight;
    T[] _bias;
    public NormalizationLayer(int dimensions) : base(0,0)
    {
        _weight = new T[dimensions];
        for(int i = 0; i < dimensions; i++)
        {
            _weight[i] = new T().Add(1);
        }

        _bias = new T[dimensions];
    }

    public override T[][,] Backwards(T[][,] dL_dP, T[][,] input, float alpha)
    {
        T[][,] dL_dPNext = new T[input.Length][,];

        for(int i = 0; i < input.Length; i++)
        {
            dL_dPNext[i] = Backwards(dL_dP[i], input[i], i, alpha);
        }
        return dL_dPNext;
    }

    T[,] Backwards(T[,] dL_dP, T[,] input, int dimension, float alpha)
    {
        int x = input.GetLength(0);
        int y = input.GetLength(1);
        int xy = x * y;

        T mean = new();
        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                mean = mean.Add(input[i, j]);
            }
        }
        mean = mean.Multiply(1f / xy);

        T sigma2 = new();

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                sigma2 = sigma2.Add(input[i, j].Subtract(mean).Pow(2));
            }
        }

        T sigma = sigma2.Add(0.001f).Pow(0.5f);

        T[,] normalized = new T[x, y];
        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                normalized[i, j] = input[i, j].Subtract(mean).Divide(sigma).ReLU().Multiply(_weight[dimension]).Add(_bias[dimension]);
            }
        }

        T dL_dW = new();
        T sum_dL_dP = new();
        T dL_dS = new();
        for (int i = 0; i < x; i++)
        {
            for(int j = 0; j < y; j++)
            {
                dL_dW = dL_dW.Add(dL_dP[i, j].Multiply(normalized[i,j]));
                sum_dL_dP = sum_dL_dP.Add(dL_dP[i, j]);
                dL_dS = dL_dS.Add(dL_dP[i, j].Add(input[i, j]));
            }
        }
        dL_dS = dL_dS.Subtract(mean.Multiply( xy));
        dL_dS = dL_dS.Multiply(sigma.Pow(-1.5f).Multiply(_weight[dimension]).Multiply(-0.5f));
        T dl_dM = sum_dL_dP.Multiply(-1).Multiply(_weight[dimension]).Divide(sigma);

        T[,] dL_dPNext = new T[x, y];

        for(int i = 0; i < x; i++)
        {
            for(int j =0; j < y; j++)
            {
                dL_dPNext[i, j] = dL_dP[i, j].Divide(sigma).Add(dL_dS.Multiply(input[i, j].Subtract(mean)).Multiply(2 / xy)).Add(dl_dM.Multiply(1 / xy));
            }
        }

        _weight[dimension] = _weight[dimension].Subtract(dL_dW.Multiply(alpha));
        _bias[dimension] = _bias[dimension].Subtract(sum_dL_dP.Multiply(alpha));

        return dL_dPNext;
    }

    public override T[][,] Forward(T[][,] input)
    {
        T[][,] output = new T[input.Length][,];

        for(int i = 0; i < input.Length; i++)
        {
            output[i] = Forward(input[i], i);
        }

        return output;
    }

    T[,] Forward(T[,] input, int dimension)
    {
        int x = input.GetLength(0);
        int y = input.GetLength(1);

        T mean = new();
        for(int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                mean = mean.Add(input[i, j]);
            }
        }
        mean = mean.Multiply(1f / x / y);

        T sigma2 = new();

        for(int i = 0; i < x; i++)
        {
            for(int j = 0; j < y; j++)
            {
                sigma2 = sigma2.Add(input[i, j].Subtract(mean).Pow(2));
            }
        }

        T sigma = sigma2.Add(0.001f).Pow(0.5f);

        T[,] output = new T[x, y];
        for(int i = 0; i < x; i++)
        {
            for(int j = 0; j < y; j++)
            {
                output[i,j] = input[i,j].Subtract(mean).Divide(sigma).ReLU().Multiply(_weight[dimension]).Add(_bias[dimension]);
            }
        }

        return output;
    }
}

