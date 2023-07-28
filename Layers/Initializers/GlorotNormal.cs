using ConvolutionalNeuralNetwork.Layers.Weighted;

namespace ConvolutionalNeuralNetwork.Layers.Initializers
{
    public class GlorotNormal : IWeightInitializer
    {
        public static GlorotNormal Instance { get; } = new GlorotNormal();

        public float GetWeight(WeightedLayer layer)
        {
            float std = MathF.Sqrt(2f / (layer.FanIn + layer.FanOut));

            return Utility.RandomGauss(0, std);
        }
    }
}
