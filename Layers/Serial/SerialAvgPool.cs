using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    /// <summary>
    /// The <see cref="SerialAvgPool"/> class is an <see cref="ISerialLayer"/> for <see cref="AveragePool"/> layers.
    /// </summary>
    public class SerialAvgPool : ISerialLayer
    {
        /// <value>The scale of the pooling filter.</value>
        public int Scale { get; init; }

        /// <summary>
        /// Initializes a new instance of the <see cref="SerialAvgPool"/> class.
        /// </summary>
        /// <param name="scale">The scale of the pooling filter.</param>
        public SerialAvgPool(int scale)
        {
            Scale = scale;
        }

        [JsonConstructor] private SerialAvgPool() { }

        /// <inheritdoc />
        public Layer Construct()
        {
            return new AveragePool(Scale);
        }

        /// <inheritdoc />
        public TensorShape Initialize(TensorShape inputShape)
        {
            return new TensorShape(inputShape.Width / Scale, inputShape.Length / Scale, inputShape.Dimensions);
        }
    }
}
