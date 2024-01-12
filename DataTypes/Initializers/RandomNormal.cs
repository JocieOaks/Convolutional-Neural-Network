using ConvolutionalNeuralNetwork.Layers.Serial;

namespace ConvolutionalNeuralNetwork.DataTypes.Initializers
{
    public class RandomNormal : IWeightInitializer
    {
        float _mean;
        float _std;

        public RandomNormal(float mean, float std)
        {
            _mean = mean;
            _std = std;
        }

        public float GetWeight(SerialWeighted layer)
        {
            return Utility.RandomGauss(_mean, _std);
        }
    }
}
