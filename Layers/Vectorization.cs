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
        [JsonProperty] private ColorTensor _tensor;
        [JsonProperty] private ColorTensor _tensorFirstMoment;
        [JsonProperty] private ColorTensor _tensorSecondMoment;

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
        public FeatureMap[,] Backwards(Vector[] vectorGradient, float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            for (int batch = 0; batch < vectorGradient.Length; batch++)
            {
                float _xy = 1f / _transposedGradients[batch, 0].Area;

                ColorVector pixelGradient = _xy * vectorGradient[batch] * _tensor;

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

                for (int y = 0; y < _tensor.Length; y++)
                {
                    for (int x = 0; x < _tensor.Width; x++)
                    {
                        Color gradient = vectorGradient[batch][x] * _vectors[batch][y] * new Color(1, 0, 0);
                        Color first = _tensorFirstMoment[x, y] = firstMomentDecay * _tensorFirstMoment[x, y] + (1 - firstMomentDecay) * gradient;
                        Color second = _tensorSecondMoment[x, y] = secondMomentDecay * _tensorSecondMoment[x, y] + (1 - secondMomentDecay) * Color.Pow(gradient, 2);
                        _tensor[x, y] -= learningRate * first / (Color.Pow(second, 0.5f) + Utility.AsymptoteErrorColor);
                    }
                }
            }

            return _transposedGradients;
        }

        /// <summary>
        /// Forward propagates through the layer calculating the output <see cref="Vector"/>.
        /// </summary>
        public Vector[] Forward()
        {
            Vector[] vectors = new Vector[_transposedInput.GetLength(0)];
            for (int i = 0; i < _transposedInput.GetLength(0); i++)
            {
                for (int j = 0; j < _transposedInput.GetLength(1); j++)
                {
                    _vectors[i][j] = _transposedInput[i, j].Average();
                }

                vectors[i] = _tensor * _vectors[i];
            }

            return vectors;
        }

        /// <summary>
        /// Initializes the <see cref="Vectorization"/> for the data set being used.
        /// </summary>
        /// <param name="transposedInputs">The previous <see cref="Layer"/>'s transposed output.</param>
        /// <param name="outGradients">The previous <see cref="Layer"/>'s inGradient.</param>
        /// <param name="vectorDimensions">The length of the output <see cref="Vector"/> after performing vectorization.</param>
        public void StartUp(FeatureMap[,] transposedInputs, int vectorDimensions)
        {
            int batchSize = transposedInputs.GetLength(0);
            int featureMapDimensions = transposedInputs.GetLength(1);

            _transposedInput = transposedInputs;
            _vectorDimensions = vectorDimensions;

            if (_tensor == null || _tensor.Width != vectorDimensions)
            {
                float variance = 2f / (3 * featureMapDimensions + vectorDimensions);
                float stdDev = MathF.Sqrt(variance);
                _tensor = ColorTensor.Random(vectorDimensions, featureMapDimensions, 0, stdDev);
                _tensorFirstMoment = new ColorTensor(_vectorDimensions, featureMapDimensions);
                _tensorSecondMoment = new ColorTensor(_vectorDimensions, featureMapDimensions);
            }

            _vectors = new ColorVector[batchSize];
            _transposedGradients = new FeatureMap[batchSize, featureMapDimensions];
            for (int i = 0; i < batchSize; i++)
            {
                _vectors[i] = new ColorVector(featureMapDimensions);
                for (int j = 0; j < featureMapDimensions; j++)
                {
                    _transposedGradients[i, j] = new FeatureMap(transposedInputs[i, j].Width, transposedInputs[i, j].Length);
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
            _tensor = ColorTensor.Random(_vectorDimensions, featureMapDimensions, 0, stdDev);
            _tensorFirstMoment = new ColorTensor(_vectorDimensions, featureMapDimensions);
            _tensorSecondMoment = new ColorTensor(_vectorDimensions, featureMapDimensions);
        }
    }
}