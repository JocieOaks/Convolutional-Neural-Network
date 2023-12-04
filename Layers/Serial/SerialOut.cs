using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialOut : ISerial
    {
        [JsonProperty] private readonly int _id;
        private readonly SerialFork _source;

        public SerialOut(SerialFork source)
        {
            _source = source;
            _id = source.ID;
        }

        [JsonConstructor] private SerialOut() { }

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

        public TensorShape Initialize(TensorShape inputShape)
        {
            return _source.OutputShape;
        }
    }
}
