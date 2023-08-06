using ConvolutionalNeuralNetwork.Layers.Serial;

namespace ConvolutionalNeuralNetwork.Layers.Initializers
{
    public interface IWeightInitializer
    {
        float GetWeight(SerialWeighted layer);
    }
}
