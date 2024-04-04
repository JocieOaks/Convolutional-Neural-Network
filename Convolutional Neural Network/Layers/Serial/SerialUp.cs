using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    /// <summary>
    /// The <see cref="SerialUp"/> class is an <see cref="ISerialLayer"/> for <see cref="Upsampling"/> layers.
    /// </summary>
    public class SerialUp : ISerialLayer
    {
        /// <value>The amount to scale up the <see cref="Tensor"/> by.</value>
        public int Scale { get; init; }

        /// <summary>
        /// Initializes a new instance of the <see cref="SerialUp"/> class.
        /// </summary>
        /// <param name="scale">The amount to scale up the <see cref="Tensor"/> by.</param>
        public SerialUp(int scale)
        {
            Scale = scale;
        }

        [JsonConstructor] private SerialUp() { }

        /// <inheritdoc />
        public Layer Construct()
        {
            return new Upsampling(Scale);
        }

        /// <inheritdoc />
        public TensorShape Initialize(TensorShape inputShape)
        {
            return new TensorShape(Scale * inputShape.Width, Scale * inputShape.Length, inputShape.Dimensions);
        }
    }
}
