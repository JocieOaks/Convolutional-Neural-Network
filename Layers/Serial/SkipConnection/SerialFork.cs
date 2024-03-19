using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.SkipConnection;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial.SkipConnection
{
    public class SerialFork : ISerialLayer
    {
        private static int s_nextID = 0;

        [JsonIgnore] public static readonly Dictionary<int, Fork> Forks = new();

        public SerialFork()
        {
            ID = s_nextID++;
        }

        public int ID { get; init; }

        [JsonIgnore] public TensorShape OutputShape { get; private set; }

        public Layer Construct()
        {
            var fork = new Fork();
            Forks[ID] = fork;
            return fork;
        }

        public TensorShape Initialize(TensorShape inputShape)
        {
            OutputShape = inputShape;
            return inputShape;
        }
    }
}
