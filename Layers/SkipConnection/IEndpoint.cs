using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers.SkipConnection
{
    public interface IEndpoint
    {
        int ID { get; }

        void Connect(Vector skipConnection, Shape skipInputShape, int id);
    }
}