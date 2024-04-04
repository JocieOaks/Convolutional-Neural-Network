using ConvolutionalNeuralNetwork.Layers.Serial.Weighted;

namespace ConvolutionalNeuralNetwork.DataTypes.Initializers
{
    /// <summary>
    /// The <see cref="Predefined"/> class is an <see cref="IWeightInitializer"/>, where the
    /// weights are set externally, as an array of weights.
    /// </summary>
    public class Predefined : IWeightInitializer
    {
        private readonly List<float> _weights;
        private int _step;

        /// <summary>
        /// Initializes a new instance of the <see cref="Predefined"/> class.
        /// </summary>
        /// <param name="weights">An array of weights, to be used as the values for <see cref="Weights"/>.</param>
        public Predefined(List<float> weights)
        {
            _weights = weights;
        }

        /// <inheritdoc />
        public float GetWeight(SerialWeighted layer)
        {
            return _weights[_step++ % _weights.Count];
        }
    }
}
