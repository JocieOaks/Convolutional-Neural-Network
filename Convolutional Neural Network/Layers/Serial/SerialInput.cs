using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    /// <summary>
    /// The <see cref="SerialInput"/> class is an <see cref="ISerialLayer"/> for <see cref="Input"/> layers.
    /// </summary>
    public class SerialInput : ISerialLayer
    {
        [JsonProperty] private TensorShape _inputShape;

        /// <summary>
        /// Initializes a new instance of the <see cref="SerialInput"/> class.
        /// </summary>
        /// <param name="inputShape">The <see cref="TensorShape"/> of the input images.</param>
        public SerialInput(TensorShape inputShape)
        {
            _inputShape = inputShape;
        }

        [JsonConstructor] private SerialInput() { }

        /// <inheritdoc />
        public Layer Construct()
        {
            return new Input(_inputShape);
        }

        /// <inheritdoc />
        public TensorShape Initialize(TensorShape inputShape)
        {
            return _inputShape;
        }
    }
}
