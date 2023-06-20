using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    [Serializable]
    public class Vectorization
    {
        [JsonProperty] private FeatureMap _matrix;
        private FeatureMap[,] _transposedGradients;
        [JsonProperty] private int _vectorDimensions;
        private ColorVector[] _vectors;

        private FeatureMap[,] _transposedInput;

        [JsonConstructor]
        public Vectorization()
        {
        }

        public string Name => "Vectorization Layer";

        public void Backwards(Vector[] vectorGradient, float learningRate)
        {
            for (int i = 0; i < vectorGradient.Length; i++)
            {
                Backwards(i, vectorGradient[i], _vectors[i], learningRate);
            }
        }

        public void Backwards(int batch, Vector vectorGradient, ColorVector vector, float learningRate)
        {
            float _xy = 1f / _transposedGradients[batch, 0].Area;

            ColorVector pixelGradient = _xy * vectorGradient * _matrix;

            for (int i = 0; i < _transposedGradients.GetLength(1); i++)
            {
                for (int j = 0; j < _transposedGradients[batch, i].Length; j++)
                {
                    for (int k = 0; k < _transposedGradients[batch, i].Width; k++)
                    {
                        _transposedGradients[batch, i][k, j] = pixelGradient[i];
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

        public Vector[] Forward()
        {
            Vector[] vectors = new Vector[_transposedInput.GetLength(0)];
            for (int i = 0; i < _transposedInput.GetLength(0); i++)
            {
                ColorVector vector = new(_transposedInput.GetLength(1));
                for (int j = 0; j < _transposedInput.GetLength(1); j++)
                {
                    vector[j] = _transposedInput[i, j].Average();
                }

                vectors[i] = _matrix * vector;
            }

            return vectors;
        }

        public void StartUp(FeatureMap[,] transposedInputs, FeatureMap[,] outGradients, int vectorDimensions)
        {
            int batchSize = transposedInputs.GetLength(0);
            int featureMapDimensions = transposedInputs.GetLength(1);

            _transposedInput = transposedInputs;
            _vectorDimensions = vectorDimensions;

            if (_matrix == null || _matrix.Width != vectorDimensions)
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

            _vectors = new ColorVector[batchSize];
            _transposedGradients = new FeatureMap[batchSize, featureMapDimensions];
            for (int i = 0; i < batchSize; i++)
            {
                _vectors[i] = new ColorVector(featureMapDimensions);
                for (int j = 0; j < featureMapDimensions; j++)
                {
                    outGradients[j, i] = _transposedGradients[i, j] = new FeatureMap(transposedInputs[i, j].Width, transposedInputs[i, j].Length);
                }
            }
        }

        public void Reset()
        {
            int featureMapDimensions = _transposedInput.GetLength(1);
            float variance = 2f / (3 * featureMapDimensions + _vectorDimensions);
            float stdDev = MathF.Sqrt(variance);
            _matrix = new FeatureMap(_vectorDimensions, featureMapDimensions);

            for (int j = 0; j < featureMapDimensions; j++)
            {
                for (int i = 0; i < _vectorDimensions; i++)
                {
                    _matrix[i, j] = Color.RandomGauss(0, stdDev);
                }
            }
        }
    }
}