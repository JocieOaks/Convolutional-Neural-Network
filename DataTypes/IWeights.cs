using ConvolutionalNeuralNetwork.Layers;
using ILGPU;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    public interface IWeights
    {
        float this[int index] { get; }

        int Length { get; }

        void DecrementLiveGradient(int decrement = 1);
        void DecrementLiveWeights(int decrement = 1);
        ArrayView<T> GradientGPU<T>() where T : unmanaged;
        void Reset(float value);
        void Reset(float mean, float stdDev);
        ArrayView<T> WeightsGPU<T>() where T : unmanaged;
    }
}