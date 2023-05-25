// See https://aka.ms/new-console-template for more information
using Newtonsoft.Json;
using System.Drawing.Drawing2D;
using System.Xml;

[Serializable]
public class VectorizationLayer
{
    [JsonProperty] readonly FeatureMap _matrix;

    public VectorizationLayer(int vectorDimensions, int featureMapDimensions)
    {
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

    public Vector Forward(FeatureMap[] input) 
    { 
        ColorVector vector = new ColorVector(input.Length);

        for(int i = 0; i < input.Length; i++)
        {
            vector[i] = input[i].Average();
        }

        return _matrix * vector;
    }

    public FeatureMap[] Backwards(FeatureMap[] input, Vector dL_dI, float learningRate)
    {
        FeatureMap[] dL_dP = new FeatureMap[input.Length];
        int x = input[0].Width;
        int y = input[0].Length;
        float _xy = 1f / (x * y);

        ColorVector dL_dPV = _xy * dL_dI * _matrix;

        for(int i = 0; i < input.Length; i++)
        {
            dL_dP[i] = new FeatureMap(x, y, dL_dPV[i]);
        }

        for (int j = 0; j < _matrix.Length; j++)
        {
            for (int i = 0; i < _matrix.Width; i++)
            {
                Color val = dL_dI[i] * input[j].Average();
                _matrix[i, j] -= learningRate * val;
            }
        }

        return dL_dP;
    }
}
