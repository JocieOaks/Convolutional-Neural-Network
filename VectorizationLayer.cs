// See https://aka.ms/new-console-template for more information
using Newtonsoft.Json;
using System.Drawing.Drawing2D;
using System.Xml;

[Serializable]
public class VectorizationLayer
{
    [JsonProperty] readonly FeatureMap _matrix;
    ColorVector[] _vectors;
    FeatureMap[][] _featureMapGradient;
    public VectorizationLayer(int vectorDimensions, FeatureMap[][] input)
    {
        int featureMapDimensions = input.Length;
        int batchSize = input[0].Length;
        float variance = 2f / (3 * featureMapDimensions + vectorDimensions);
        float stdDev = MathF.Sqrt(variance);
        _matrix = new FeatureMap(vectorDimensions, featureMapDimensions);

        _vectors = new ColorVector[batchSize];
        _featureMapGradient = new FeatureMap[batchSize][];

        for(int i = 0; i < batchSize; i++)
        {
            _vectors[i] = new ColorVector(featureMapDimensions);
            _featureMapGradient[i] = new FeatureMap[featureMapDimensions];
            for(int j = 0; j < featureMapDimensions; j++)
            {
                _featureMapGradient[i][j] = new FeatureMap(input[j][i].Width, input[j][i].Length);
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

    public FeatureMap[][] Backwards(Vector[] vectorGradient, float learningRate)
    {
        for(int i = 0; i < vectorGradient.Length; i++)
        {
            Backwards(_featureMapGradient[i], vectorGradient[i], _vectors[i], learningRate);
        }

        return _featureMapGradient;
    }

    public FeatureMap[] Backwards(FeatureMap[] featureMapGradient, Vector vectorGradient, ColorVector vector, float learningRate)
    {
        float _xy = 1f / featureMapGradient[0].Area;

        ColorVector pixelGradient = _xy * vectorGradient * _matrix;

        for(int i = 0; i < featureMapGradient.Length; i++)
        {
            for(int j = 0; j < featureMapGradient[i].Length; j++)
            {
                for(int k = 0; k < featureMapGradient[i].Width; k++)
                {
                    featureMapGradient[i][k, j] = pixelGradient[i];
                }
            }
        }

        for (int j = 0; j < _matrix.Length; j++)
        {
            for (int i = 0; i < _matrix.Width; i++)
            {
                Color val = vectorGradient[i] * vector[j];
                _matrix[i, j] -= learningRate * val;
            }
        }

        return featureMapGradient;
    }
}
