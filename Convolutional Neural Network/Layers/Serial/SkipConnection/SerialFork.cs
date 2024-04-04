using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.SkipConnection;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial.SkipConnection
{
    /// <summary>
    /// The <see cref="SerialFork"/> class is an <see cref="ISerialLayer"/> for <see cref="Fork"/> layers.
    /// </summary>
    public class SerialFork : ISerialLayer
    {
        private static int s_nextID;

        /// <summary>
        /// Initializes a new instance of the <see cref="SerialFork"/> class.
        /// </summary>
        public SerialFork()
        {
            ID = s_nextID++;
        }

        /// <value>A dictionary of <see cref="Fork"/>s used to search by ID to construct their corresponding <see cref="IForkEndpoint"/>s.</value>
        [JsonIgnore] public static Dictionary<int, Fork> Forks { get; } = new();

        /// <value>An identifier used to connect <see cref="Fork"/>s and <see cref="IForkEndpoint"/>s.</value>
        public int ID { get; init; }

        /// <value>The <see cref="TensorShape"/> of the forked output copied to the corresponding <see cref="IForkEndpoint"/>.</value>
        [JsonIgnore] public TensorShape OutputShape { get; private set; }

        /// <inheritdoc />
        public Layer Construct()
        {
            var fork = new Fork();
            Forks[ID] = fork;
            return fork;
        }

        /// <inheritdoc />
        public TensorShape Initialize(TensorShape inputShape)
        {
            OutputShape = inputShape;
            return inputShape;
        }
    }
}
