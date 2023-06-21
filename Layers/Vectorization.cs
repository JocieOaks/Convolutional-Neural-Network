using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="Vectorization"/> class is the last layer of a <see cref="Networks.Discriminator"/> that converts the set of 
    /// <see cref="FeatureMap"/>s into a <see cref="Vector"/>.
    /// </summary>
    [Serializable]
    public class Vectorization
    {
        [JsonProperty] private FeatureMap _matrix;
        private FeatureMap[,] _transposedGradients;
        [JsonProperty] private int _vectorDimensions;
        private ColorVector[] _vectors;

        private FeatureMap[,] _transposedInput;

        /// <value>The name of the layer, used for logging.</value>
        public string Name => "Vectorization Layer";

        /// <summary>
        /// Backpropagates through the <see cref="Layer"/> updating any layer weights, and calculating the outgoing gradient that is
        /// shared with the previous layer.
        /// </summary>
        /// <param name="learningRate">The overall learning rate for the layer updates.</param>
        public void Backwards(Vector[] vectorGradient, float learningRate)
        {
            for (int batch = 0; batch < vectorGradient.Length; batch++)
            {
                float _xy = 1f / _transposedGradients[batch, 0].Area;

                ColorVector pixelGradient = _xy * vectorGradient[batch] * _matrix;

                for (int dimension = 0; dimension < _transposedGradients.GetLength(1); dimension++)
                {
                    for (int y = 0; y < _transposedGradients[batch, dimension].Length; y++)
                    {
                        for (int x = 0; x < _transposedGradients[batch, dimension].Width; x++)
                        {
                            _transposedGradients[batch, dimension][x, y] = pixelGradient[dimension];
                        }
                    }
                }

                for (int y = 0; y < _matrix.Length; y++)
                {
                    for (int x = 0; x < _matrix.Width; x++)
                    {
                        Color val = vectorGradient[batch][x] * _vectors[batch][y];
                        _matrix[x, y] -= learningRate * val;
                    }
                }
            }
        }

        /// <summary>
        /// Forward propagates through the layer calculating the output <see cref="Vector"/>.
        /// </summary>
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

        /// <summary>
        /// Initializes the <see cref="Vectorization"/> for the data set being used.
        /// </summary>
        /// <param name="transposedInputs">The previous <see cref="Layer"/>'s transposed output.</param>
        /// <param name="outGradients">The previous <see cref="Layer"/>'s inGradient.</param>
        /// <param name="vectorDimensions">The length of the output <see cref="Vector"/> after performing vectorization.</param>
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

        /// <summary>
        /// Reset's the <see cref="Vectorization"/> layer to random initial weights.
        /// </summary>
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