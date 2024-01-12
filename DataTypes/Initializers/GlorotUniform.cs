using ConvolutionalNeuralNetwork.Layers.Serial;

namespace ConvolutionalNeuralNetwork.DataTypes.Initializers
{
    public class GlorotUniform : IWeightInitializer
    {
        public static GlorotUniform Instance { get; } = new GlorotUniform();

        public float GetWeight(SerialWeighted layer)
        {
            float limit = MathF.Sqrt(6f / (layer.FanIn + layer.FanOut));

            return (Utility.Random.NextSingle() * 2 - 1) * limit;
        }
    }
}
