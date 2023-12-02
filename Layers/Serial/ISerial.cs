using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public interface ISerial
    {
        Shape Initialize(Shape inputShape);

        Layer Construct();
    }
}
