using ConvolutionalNeuralNetwork.Layers.Serial.Weighted;

namespace ConvolutionalNeuralNetwork.DataTypes.Initializers
{
    /// <summary>
    /// The <see cref="RandomNormal"/> class is an <see cref="IWeightInitializer"/> that returns values in a uniform distribution,
    /// based on the specified max and min values.
    /// </summary>
    public class RandomUniform : IWeightInitializer
    {
        private readonly float _min;
        private readonly float _delta;

        /// <summary>
        /// Initializes a new instance of the <see cref="RandomUniform"/> class.
        /// </summary>
        /// <param name="max">The maximum value for weights.</param>
        /// <param name="min">The minimum value for weights. Defaults to NaN, in which case min will be negative <paramref name="max"/>.</param>
        /// <exception cref="ArgumentException">Thrown is <paramref name="max"/> is less than <paramref name="min"/>.</exception>
        public RandomUniform(float max, float min = float.NaN)
        {
            if (float.IsNaN(min))
            {
                _min = -max;
            }
            else
            {
                _min = min;
            }
            if (max < min)
            {
                throw new ArgumentException("Max is less than min.");
            }

            _delta = max - _min;
        }

        /// <inheritdoc />
        public float GetWeight(SerialWeighted layer)
        {
            return Utility.Random.NextSingle() * _delta + _min;
        }
    }
}
