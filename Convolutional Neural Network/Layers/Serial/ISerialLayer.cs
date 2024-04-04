using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    /// <summary>
    /// The <see cref="ISerialLayer"/> interface is used for creating serializable forms of <see cref="Layer"/>s.
    /// Serial layers function as a "blueprint" for its <see cref="Layer"/>.
    /// </summary>
    public interface ISerialLayer
    {
        /// <summary>
        /// Initializes the data to be used be the corresponding <see cref="Layer"/>.
        /// </summary>
        /// <param name="inputShape">The <see cref="TensorShape"/> of the input <see cref="Tensor"/> of the layer.</param>
        /// <returns>The <see cref="TensorShape"/> of the output <see cref="Tensor"/> of the layer.</returns>
        TensorShape Initialize(TensorShape inputShape);

        /// <summary>
        /// Constructs the corresponding <see cref="Layer"/> to this <see cref="ISerialLayer"/>.
        /// </summary>
        Layer Construct();
    }
}
