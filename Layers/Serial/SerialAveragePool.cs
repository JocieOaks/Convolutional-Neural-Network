using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialAveragePool : ISerial
    {
        public int Scale { get; init; }

        public SerialAveragePool(int scale)
        {
            Scale = scale;
        }

        [JsonConstructor] private SerialAveragePool() { }

        public Layer Construct()
        {
            return new AveragePool(Scale);
        }

        public Shape Initialize(Shape inputShape)
        {
            return new Shape(inputShape.Width / Scale, inputShape.Length / Scale, inputShape.Dimensions);
        }
    }
}
