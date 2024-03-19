using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    /// <summary>
    /// The <see cref="SerialReshape"/> class is an <see cref="ISerialLayer"/> for <see cref="Reshape"/> layers.
    /// </summary>
    public class SerialReshape : ISerialLayer
    {
        [JsonProperty] private TensorShape _outputShape;

        /// <summary>
        /// Initializes a new instance of the <see cref="SerialReshape"/> class.
        /// </summary>
        /// <param name="outputShape">The <see cref="TensorShape"/> of the layer's output.</param>
        public SerialReshape(TensorShape outputShape)
        {
            _outputShape = outputShape;
        }

        [JsonConstructor] private SerialReshape() { }

        /// <inheritdoc />
        public Layer Construct()
        {
            return new Reshape(_outputShape);
        }

        /// <inheritdoc />
        public TensorShape Initialize(TensorShape inputShape)
        {
            if (inputShape.Volume != _outputShape.Volume)
            {
                throw new ArgumentException("Input and output shapes have different lengths.");
            }

            return _outputShape;
        }
    }
}
