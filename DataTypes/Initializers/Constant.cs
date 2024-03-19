using ConvolutionalNeuralNetwork.Layers.Serial.Weighted;

namespace ConvolutionalNeuralNetwork.DataTypes.Initializers
{
    /// <summary>
    /// The <see cref="Constant"/> class is an <see cref="IWeightInitializer"/> that always returns the same value,
    /// specified at construction.
    /// </summary>
    public class Constant : IWeightInitializer
    {
        private readonly float _constant;

        /// <summary>
        /// Initializes a new instance of the <see cref="Constant"/> class.
        /// </summary>
        /// <param name="constant">The constant to be returned.</param>
        public Constant(float constant)
        {
            _constant = constant;
        }

        /// <inheritdoc />
        public float GetWeight(SerialWeighted layer)
        {
            return _constant;
        }
    }
}
