using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.Activations;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    /// <summary>
    /// The <see cref="Activation"/> enum is used to designate which activation <see cref="Layer"/> is to be constructed.
    /// </summary>
    public enum Activation
    {
        /// <value>Indicates no activation <see cref="Layer"/> is to be used.</value>
        None,
        /// <value>Indicates <see cref="Activations.ReLU"/>.</value>
        ReLU,
        /// <value>Indicates <see cref="Activations.LeakyReLU"/>.</value>
        LeakyReLU,
        /// <value>Indicates <see cref="Activations.Sigmoid"/>.</value>
        Sigmoid,
        /// <value>Indicates <see cref="HyperTan"/>.</value>
        HyperbolicTangent,
        /// <value>Indicates <see cref="Activations.Proportion"/>.</value>
        Proportion,
        /// <value>Indicates <see cref="Activations.Dropout"/>.</value>
        Dropout
    }

    /// <summary>
    /// The <see cref="SerialActivation"/> class is an <see cref="ISerialLayer"/> that is used for creating activation <see cref="Layer"/>s.
    /// </summary>
    public class SerialActivation : ISerialLayer
    {
        /// <value>The activation <see cref="Layer"/> this <see cref="SerialActivation"/> is used for.</value>
        public Activation Activation { get; init; } = Activation.ReLU;

        public float ActivationValues { get; init; }

        /// <inheritdoc />
        public Layer Construct()
        {
            return Activation switch
            {
                Activation.Sigmoid => new Sigmoid(),
                Activation.HyperbolicTangent => new HyperTan(),
                Activation.LeakyReLU => new LeakyReLU(ActivationValues),
                Activation.Proportion => new Proportion(),
                Activation.Dropout => new Dropout(ActivationValues),
                _ => new ReLU()
            };
        }

        /// <inheritdoc />
        public TensorShape Initialize(TensorShape inputShape)
        {
            return inputShape;
        }
    }
}
