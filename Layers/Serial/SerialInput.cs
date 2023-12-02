using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialInput : ISerial
    {
        [JsonProperty] Shape _inputShape;

        public SerialInput(Shape inputShape)
        {
            _inputShape = inputShape;
        }

        [JsonConstructor] public SerialInput() { }

        public Layer Construct()
        {
            return new InputLayer(_inputShape);
        }

        public Shape Initialize(Shape inputShape)
        {
            return _inputShape;
        }
    }
}
