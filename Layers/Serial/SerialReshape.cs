using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialReshape : ISerial
    {
        [JsonProperty] private TensorShape _outputShape;

        public SerialReshape(TensorShape outputShape)
        {
            _outputShape = outputShape;
        }

        [JsonConstructor] private SerialReshape() { }

        public Layer Construct()
        {
            return new Reshape(_outputShape);
        }

        public TensorShape Initialize(TensorShape inputShape)
        {
            int inputLength = inputShape.Volume;

            int outputLength = _outputShape.Volume;

            if (inputLength != outputLength)
            {
                throw new ArgumentException("Input and output shapes have different lengths.");
            }

            return _outputShape;
        }
    }
}
