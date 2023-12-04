using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.Augmentations;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public enum Augmentation
    {
        Cutout,
        Translation
    }

    public class SerialAugmentation : ISerial
    {
        public Augmentation Augmentation { get; init; }

        public Layer Construct()
        {
            return Augmentation switch
            {
                Augmentation.Cutout => new Cutout(),
                Augmentation.Translation => new Translation(),
                _ => throw new ArgumentException()
            };
        }

        public TensorShape Initialize(TensorShape inputShape)
        {
            return inputShape;
        }
    }
}
