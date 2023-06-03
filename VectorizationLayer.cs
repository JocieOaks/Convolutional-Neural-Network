// See https://aka.ms/new-console-template for more information
using Newtonsoft.Json;
using System.Drawing.Drawing2D;
using System.Numerics;
using System.Xml;

[Serializable]
public class VectorizationLayer
{
    [JsonProperty] readonly FeatureMap _matrix;
    ColorVector[] _vectors;
    FeatureMap[,] _transposedGradient;
    public VectorizationLayer(int vectorDimensions, FeatureMap[,] input)
    {
        StartUp(input);

        int featureMapDimensions = input.GetLength(0);
        
        float variance = 2f / (3 * featureMapDimensions + vectorDimensions);
        float stdDev = MathF.Sqrt(variance);
        _matrix = new FeatureMap(vectorDimensions, featureMapDimensions);

        for (int j = 0; j < featureMapDimensions; j++)
        {
            for (int i = 0; i < vectorDimensions; i++)
            {
                _matrix[i, j] = Color.RandomGauss(0, stdDev);
            }
        }
    }

    public void StartUp(FeatureMap[,] input)
    {
        int featureMapDimensions = input.GetLength(0);
        int batchSize = input.GetLength(1);

        _vectors = new ColorVector[batchSize];
        _transposedGradient = new FeatureMap[batchSize, featureMapDimensions];

        for (int i = 0; i < batchSize; i++)
        {
            _vectors[i] = new ColorVector(featureMapDimensions);
            for (int j = 0; j < featureMapDimensions; j++)
            {
                _transposedGradient[i, j] = new FeatureMap(input[j, i].Width, input[j, i].Length);
            }
        }
    }

    public string Name => "Vectorization Layer";

    public Vector[] Forward(FeatureMap[,] inputTransposed)
    {
        Vector[] vectors = new Vector[inputTransposed.GetLength(0)];
        for(int i = 0; i < inputTransposed.GetLength(0); i++)
        {
            ColorVector vector = new ColorVector(inputTransposed.GetLength(1));
            for (int j = 0; j < inputTransposed.GetLength(1); j++)
            {
                vector[j] = inputTransposed[i, j].Average();
            }

            vectors[i] = _matrix * vector;
        }

        return vectors;
    }

    public FeatureMap[,] Backwards(Vector[] vectorGradient, float learningRate)
    {
        for(int i = 0; i < vectorGradient.Length; i++)
        {
            Backwards(i, vectorGradient[i], _vectors[i], learningRate);
        }

        return _transposedGradient;
    }

    public void Backwards(int batch, Vector vectorGradient, ColorVector vector, float learningRate)
    {
        float _xy = 1f / _transposedGradient[batch, 0].Area;

        ColorVector pixelGradient = _xy * vectorGradient * _matrix;

        for(int i = 0; i < _transposedGradient.GetLength(1); i++)
        {
            for(int j = 0; j < _transposedGradient[batch, i].Length; j++)
            {
                for(int k = 0; k < _transposedGradient[batch, i].Width; k++)
                {
                    _transposedGradient[batch, i][k, j] = pixelGradient[i];
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
    }
}
