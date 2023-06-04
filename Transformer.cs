// See https://aka.ms/new-console-template for more information
using Newtonsoft.Json;


[Serializable]
public class Transformer
{
    [JsonProperty] private readonly float[,] _boolMatrix;
    [JsonProperty] private readonly float[,] _floatMatrix;

    public Transformer(int bools, int floats, int outputs)
    {
        float variance = 2f / (bools + floats + outputs);
        float stdDev = MathF.Sqrt(variance);

        _boolMatrix = new float[outputs, bools];
        for (int i = 0; i < outputs; i++)
        {
            for (int j = 0; j < bools; j++)
            {
                _boolMatrix[i, j] = ConvolutionalNeuralNetwork.RandomGauss(0, stdDev);
            }
        }

        _floatMatrix = new float[outputs, floats];
        for (int i = 0; i < outputs; i++)
        {
            for (int j = 0; j < floats; j++)
            {
                _floatMatrix[i, j] = (float)ConvolutionalNeuralNetwork.Random.NextDouble() - 0.5f;
            }
        }
    }

    [JsonConstructor]
    private Transformer()
    {
    }

    public Vector Forward(bool[] bools, float[] floats)
    {
        Vector vector = _floatMatrix * new Vector(floats);

        for (int i = 0; i < _boolMatrix.GetLength(0); i++)
        {
            for (int j = 0; j < _boolMatrix.GetLength(1); j++)
            {
                if (bools[j])
                    vector[i] += _boolMatrix[i, j];
            }
        }

        return vector;
    }

    public void Backwards(bool[] bools, float[] floats, Vector descriptionGradient, float learningRate)
    {
        for (int i = 0; i < _floatMatrix.GetLength(0); i++)
        {
            for (int j = 0; j < _floatMatrix.GetLength(1); j++)
            {
                _floatMatrix[i, j] -= learningRate * descriptionGradient[i] * floats[j];
            }
        }

        for (int i = 0; i < _boolMatrix.GetLength(0); i++)
        {
            for (int j = 0; j < _boolMatrix.GetLength(1); j++)
            {
                if (bools[j])
                    _boolMatrix[i, j] -= learningRate * descriptionGradient[i];
            }
        }
    }
}