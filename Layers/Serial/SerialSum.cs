using ConvolutionalNeuralNetwork.DataTypes;
using System.Text.Json.Serialization;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    /// <summary>
    /// The <see cref="SerialSum"/> class is an <see cref="ISerialLayer"/> for <see cref="Summation"/> layers.
    /// </summary>
    public class SerialSum : ISerialLayer
    {
        /// <value>The dimension of the output <see cref="Tensor"/>.</value>
        public int OutputDimensions { get; init; }

        /// <summary>
        /// Initializes a new instance of the <see cref="SerialSum"/> class.
        /// </summary>
        /// <param name="outputDimensions">The dimension of the output <see cref="Tensor"/>.</param>
        public SerialSum(int outputDimensions)
        {
            OutputDimensions = outputDimensions;
        }

        [JsonConstructor] private SerialSum() { }

        /// <inheritdoc />
        public Layer Construct()
        {
            return new Summation(OutputDimensions);
        }

        /// <inheritdoc />
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
