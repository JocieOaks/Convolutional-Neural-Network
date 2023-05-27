// See https://aka.ms/new-console-template for more information
using Newtonsoft.Json;
using System.Drawing.Drawing2D;
using System.Xml;

[Serializable]
public class VectorizationLayer
{
    [JsonProperty] readonly FeatureMap _matrix;
    ColorVector[] _vectors;
    FeatureMap[][] _dL_dP;
    public VectorizationLayer(int vectorDimensions, FeatureMap[][] input)
    {
        int featureMapDimensions = input.Length;
        int batchSize = input[0].Length;
        float variance = 2f / (3 * featureMapDimensions + vectorDimensions);
        float stdDev = MathF.Sqrt(variance);
        _matrix = new FeatureMap(vectorDimensions, featureMapDimensions);

        _vectors = new ColorVector[batchSize];
        _dL_dP = new FeatureMap[batchSize][];

        for(int i = 0; i < batchSize; i++)
        {
            _vectors[i] = new ColorVector(featureMapDimensions);
            _dL_dP[i] = new FeatureMap[featureMapDimensions];
            for(int j = 0; j < featureMapDimensions; j++)
            {
                _dL_dP[i][j] = new FeatureMap(input[j][i].Width, input[j][i].Length);
            }
        }

        for (int j = 0; j < featureMapDimensions; j++)
        {
            for (int i = 0; i < vectorDimensions; i++)
            {
                _matrix[i, j] = Color.RandomGauss(0, stdDev);
            }
        }
    }

    public Vector[] Forward(FeatureMap[][] input)
    {
        Vector[] vectors = new Vector[input.Length];
        for(int i = 0; i < input.Length; i++)
        {
            vectors[i] = Forward(input[i], _vectors[i]);
        }

        return vectors;
    }

    public Vector Forward(FeatureMap[] input, ColorVector vector) 
    { 
        for(int i = 0; i < input.Length; i++)
        {
            vector[i] = input[i].Average();
        }

        return _matrix * vector;
    }

    public FeatureMap[][] Backwards(Vector[] dL_dI, float learningRate)
    {
        for(int i = 0; i < dL_dI.Length; i++)
        {
            Backwards(_dL_dP[i], dL_dI[i], _vectors[i], learningRate);
        }

        return _dL_dP;
    }

    public FeatureMap[] Backwards(FeatureMap[] dL_dP, Vector dL_dI, ColorVector vector, float learningRate)
    {
        float _xy = 1f / dL_dP[0].Area;

        ColorVector dL_dPV = _xy * dL_dI * _matrix;

        for(int i = 0; i < dL_dP.Length; i++)
        {
            for(int j = 0; j < dL_dP[i].Length; j++)
            {
                for(int k = 0; k < dL_dP[i].Width; k++)
                {
                    dL_dP[i][k, j] = dL_dPV[i];
                }
            }
        }

        for (int j = 0; j < _matrix.Length; j++)
        {
            for (int i = 0; i < _matrix.Width; i++)
            {
                Color val = dL_dI[i] * vector[j];
                _matrix[i, j] -= learningRate * val;
            }
        }

        return dL_dP;
    }
}
