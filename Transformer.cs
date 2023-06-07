// See https://aka.ms/new-console-template for more information
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;


[Serializable]
public class Transformer
{
    [JsonProperty] private float[,] _boolMatrix;
    [JsonProperty] private float[,] _floatMatrix;
    [JsonProperty] private int _vectorDimensions;
    public Transformer(int vectorDimensions)
    {
        _vectorDimensions = vectorDimensions;
    }

    public void ChangeVectorDimensions(int vectorDimensions)
    {
        _vectorDimensions = vectorDimensions;
    }

    public bool Startup(int bools, int floats)
    {
        bool initialize = false;
        float variance = 2f / (bools + floats + _vectorDimensions);
        float stdDev = MathF.Sqrt(variance);

        if (_boolMatrix == null || _boolMatrix.GetLength(0) != _vectorDimensions || _boolMatrix.GetLength(1) != bools)
        {
            initialize = true;
            _boolMatrix = new float[_vectorDimensions, bools];
            for (int i = 0; i < _vectorDimensions; i++)
            {
                for (int j = 0; j < bools; j++)
                {
                    _boolMatrix[i, j] = ConvolutionalNeuralNetwork.RandomGauss(0, stdDev);
                }
            }
        }
        if (_floatMatrix == null || _floatMatrix.GetLength(0) != _vectorDimensions || _floatMatrix.GetLength(1) != floats)
        {
            initialize = true;
            _floatMatrix = new float[_vectorDimensions, floats];
            for (int i = 0; i < _vectorDimensions; i++)
            {
                for (int j = 0; j < floats; j++)
                {
                    _floatMatrix[i, j] = ConvolutionalNeuralNetwork.RandomGauss(0, stdDev);
                }
            }
        }
        return initialize;
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
                _floatMatrix[i, j] -= learningRate * 5 * descriptionGradient[i] * floats[j];
            }
        }

        for (int i = 0; i < _boolMatrix.GetLength(0); i++)
        {
            for (int j = 0; j < _boolMatrix.GetLength(1); j++)
            {
                if (bools[j])
                    _boolMatrix[i, j] -= learningRate * 5 * descriptionGradient[i];
            }
        }
    }
}