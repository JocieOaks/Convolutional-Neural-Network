using ConvolutionalNeuralNetwork.Layers.Serial.Weighted;

namespace ConvolutionalNeuralNetwork.DataTypes.Initializers
{
    /// <summary>
    /// The <see cref="RandomNormal"/> class is an <see cref="IWeightInitializer"/> that returns values in a normal distribution,
    /// based on a specified mean and standard deviation.
    /// </summary>
    public class RandomNormal : IWeightInitializer
    {
        private readonly float _mean;
        private readonly float _std;

        /// <summary>
        /// Initializes a new instance of the <see cref="RandomNormal"/> class.
        /// </summary>
        /// <param name="mean">The mean of the normal distribution.</param>
        /// <param name="std">The standard deviation of the normal distribution.</param>
        public RandomNormal(float mean, float std)
        {
            _mean = mean;
            _std = std;
        }

        /// <inheritdoc />
        public float GetWeight(SerialWeighted layer)
        {
            return Utility.RandomGauss(_mean, _std);
        }
    }
}
