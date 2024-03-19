using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers.SkipConnection
{
    /// <summary>
    /// The <see cref="IEndpoint"/> interface is for <see cref="Layer"/>s that receive a copy of a <see cref="Tensor"/>
    /// from a <see cref="Fork"/> for forming skip connections.
    /// </summary>
    public interface IEndpoint
    {
        /// <summary>
        /// Connects the <see cref="IEndpoint"/> with its <see cref="Fork"/> sharing the <see cref="Tensor"/>s
        /// between them.
        /// </summary>
        /// <param name="skipConnection">A <see cref="Vector"/> that will be used to store a copy of the
        /// <see cref="Tensor"/> from the corresponding <see cref="Fork"/>.</param>
        /// <param name="skipInputShape">The <see cref="TensorShape"/> of the <see cref="Tensor"/> copied from <see cref="Fork"/>.</param>
        void Connect(Vector skipConnection, TensorShape skipInputShape);
    }
}