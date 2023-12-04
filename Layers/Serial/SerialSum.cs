using ConvolutionalNeuralNetwork.DataTypes;
using System.Text.Json.Serialization;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialSum : ISerial
    {
        public int OutputDimensions { get; init; }

        public SerialSum(int outputDimensions)
        {
            OutputDimensions = outputDimensions;
        }

        [JsonConstructor] private SerialSum() { }

        public Layer Construct()
        {
            return new Summation(OutputDimensions);
        }

        public TensorShape Initialize(TensorShape inputShape)
        {
            if(inputShape.Dimensions %  OutputDimensions != 0)
            {
                throw new ArgumentException("Input cannot be summed evenly over output dimensions.");
            }

            return new TensorShape(inputShape.Width, inputShape.Length, OutputDimensions);
        }
    }
}
