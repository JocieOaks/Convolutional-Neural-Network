using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialWarp : ISerial
    {
        public Layer Construct()
        {
            return new Warp();
        }

        public TensorShape Initialize(TensorShape inputShape)
        {
            return new TensorShape(inputShape.Width, inputShape.Length, inputShape.Dimensions - 2);
        }
    }
}
