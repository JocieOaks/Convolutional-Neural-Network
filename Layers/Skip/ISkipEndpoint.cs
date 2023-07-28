using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers.Skip
{
    public interface ISkipEndpoint
    {
        int ID { get; }

        void Connect(Vector skipConnection, Shape skipInputShape, int id);
    }
}