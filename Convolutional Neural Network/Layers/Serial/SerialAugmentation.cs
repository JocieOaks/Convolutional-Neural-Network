using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.Augmentations;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    /// <summary>
    /// The <see cref="Augmentation"/> enum is used to designate which augmentation <see cref="Layer"/> is to be constructed.
    /// </summary>
    public enum Augmentation
    {
        /// <value>Indicates <see cref="Augmentations.Cutout"/></value>
        Cutout,
        /// <value>Indicates <see cref="Augmentations.Translation"/></value>
        Translation
    }

    /// <summary>
    /// The <see cref="SerialAugmentation"/> class is an <see cref="ISerialLayer"/> that is used for creating augmentation <see cref="Layer"/>s.
    /// </summary>
    public class SerialAugmentation : ISerialLayer
    {
        /// <value>The augmentation <see cref="Layer"/> this <see cref="SerialAugmentation"/> is used for.</value>
        public Augmentation Augmentation { get; init; }

        /// <inheritdoc />
        public Layer Construct()
        {
            return Augmentation switch
            {
                Augmentation.Cutout => new Cutout(),
                Augmentation.Translation => new Translation(),
                _ => throw new ArgumentException()
            };
        }

        /// <inheritdoc />
        public TensorShape Initialize(TensorShape inputShape)
        {
            return inputShape;
        }
    }
}
