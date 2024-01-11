using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialInput : ISerial
    {
        [JsonProperty] TensorShape _inputShape;

        public SerialInput(TensorShape inputShape)
        {
            _inputShape = inputShape;
        }

        [JsonConstructor] public SerialInput() { }

        public Layer Construct()
        {
            return new Input(_inputShape);
        }

        public TensorShape Initialize(TensorShape inputShape)
        {
            return _inputShape;
        }
    }
}
