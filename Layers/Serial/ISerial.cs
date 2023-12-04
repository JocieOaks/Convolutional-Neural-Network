using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public interface ISerial
    {
        TensorShape Initialize(TensorShape inputShape);

        Layer Construct();
    }
}
