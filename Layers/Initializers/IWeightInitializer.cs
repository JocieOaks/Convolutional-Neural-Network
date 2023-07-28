using ConvolutionalNeuralNetwork.Layers.Weighted;

namespace ConvolutionalNeuralNetwork.Layers.Initializers
{
    public interface IWeightInitializer
    {
        float GetWeight(WeightedLayer layer);
    }
}
