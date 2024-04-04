using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.SkipConnection;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial.SkipConnection
{
    /// <summary>
    /// The <see cref="SerialOut"/> class is an <see cref="ISerialLayer"/> for <see cref="Out"/> layers.
    /// </summary>
    public class SerialOut : ISerialLayer
    {
        [JsonProperty] private readonly int _id;
        private readonly SerialFork _source;

        /// <summary>
        /// Initializes a new instance of the <see cref="SerialOut"/> class.
        /// </summary>
        /// <param name="source">The <see cref="SerialFork"/> that connects to this <see cref="SerialOut"/>.</param>
        public SerialOut(SerialFork source)
        {
            _source = source;
            _id = source.ID;
        }

        [JsonConstructor] private SerialOut() { }

        /// <inheritdoc />
        public Layer Construct()
        {
            if (SerialFork.Forks.TryGetValue(_id, out var split))
            {
                return split.GetOutLayer();
            }
            else
            {
                throw new Exception("Skip connection fork cannot be found.");
            }
        }

        /// <inheritdoc />
        public TensorShape Initialize(TensorShape inputShape)
        {
            return _source.OutputShape;
        }
    }
}
