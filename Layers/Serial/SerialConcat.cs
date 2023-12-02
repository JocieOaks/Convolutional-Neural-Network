using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialConcat : ISerial
    {
        [JsonProperty] private readonly int _id;
        private readonly SerialFork _source;

        public SerialConcat(SerialFork source)
        {
            _source = source;
            _id = source.ID;
        }

        [JsonConstructor] SerialConcat() { }

        public Layer Construct()
        {
            if (SerialFork.Forks.TryGetValue(_id, out var split))
            {
                return split.GetConcatenationLayer();
            }
            else
            {
                throw new Exception("Skip connection fork cannot be found.");
            }
        }

        public Shape Initialize(Shape inputShape)
        {
            if (inputShape.Area != _source.OutputShape.Area)
            {
                throw new ArgumentException("Input shapes do not match.");
            }

            return new Shape(inputShape.Width, inputShape.Length, inputShape.Dimensions + _source.OutputShape.Dimensions);
        }
    }
}
