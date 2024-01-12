using ConvolutionalNeuralNetwork.Layers.Serial;

namespace ConvolutionalNeuralNetwork.DataTypes.Initializers
{
    public interface IWeightInitializer
    {
        float GetWeight(SerialWeighted layer);
    }
}
