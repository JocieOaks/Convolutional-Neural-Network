using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialUp : ISerial
    {
        public int Scale { get; init; }

        public SerialUp(int scale)
        {
            Scale = scale;
        }

        [JsonConstructor] private SerialUp() { }

        public Layer Construct()
        {
            return new Upsampling(Scale);
        }

        public TensorShape Initialize(TensorShape inputShape)
        {
            return new TensorShape(Scale * inputShape.Width, Scale * inputShape.Length, inputShape.Dimensions);
        }
    }
}
