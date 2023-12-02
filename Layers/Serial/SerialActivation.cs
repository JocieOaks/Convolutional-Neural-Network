using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.Activations;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public enum Activation
    {
        None,
        ReLU,
        Sigmoid,
        HyperbolicTangent
    }

    public class SerialActivation : ISerial
    {
        public Activation Activation { get; init; } = Activation.ReLU;

        public Layer Construct()
        {
            return Activation switch
            {
                Activation.Sigmoid => new Sigmoid(),
                Activation.HyperbolicTangent => new HyperTan(),
                _ => new ReLUActivation()
            };
        }

        public Shape Initialize(Shape inputShape)
        {
            return inputShape;
        }
    }
}
