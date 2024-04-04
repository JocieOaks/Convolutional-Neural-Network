using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.SkipConnection;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial.SkipConnection
{
    /// <summary>
    /// The <see cref="SerialConcat"/> class is an <see cref="ISerialLayer"/> for <see cref="Concatenate"/> layers.
    /// </summary>
    public class SerialConcat : ISerialLayer
    {
        [JsonProperty] private readonly int _id;
        private readonly SerialFork _source;

        /// <summary>
        /// Initializes a new instance of the <see cref="SerialConcat"/> class.
        /// </summary>
        /// <param name="source">The <see cref="SerialFork"/> that connects to this <see cref="SerialConcat"/>.</param>
        public SerialConcat(SerialFork source)
        {
            _source = source;
            _id = source.ID;
        }

        [JsonConstructor] SerialConcat() { }

        /// <inheritdoc />
        public Layer Construct()
        {
            if (SerialFork.Forks.TryGetValue(_id, out var split))
            {
                return split.GetConcatenationLayer();
            }

            throw new Exception("Skip connection fork cannot be found.");
        }

        /// <inheritdoc />
        public TensorShape Initialize(TensorShape inputShape)
        {
            if (inputShape.Area != _source.OutputShape.Area)
            {
                throw new ArgumentException("Input shapes do not match.");
            }

            return new TensorShape(inputShape.Width, inputShape.Length, inputShape.Dimensions + _source.OutputShape.Dimensions);
        }
    }
}
