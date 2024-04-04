using ConvolutionalNeuralNetwork.Layers.Serial.Weighted;

namespace ConvolutionalNeuralNetwork.DataTypes.Initializers
{
    /// <summary>
    /// The <see cref="GlorotNormal"/> class is an <see cref="IWeightInitializer"/> that returns values in a normal distribution,
    /// based on the fan in/out of the <see cref="SerialWeighted"/> layer.
    /// </summary>
    public class GlorotNormal : IWeightInitializer
    {
        /// <value>A static instance of the <see cref="GlorotNormal"/> class.</value>
        public static GlorotNormal Instance { get; } = new ();

        /// <inheritdoc />
        public float GetWeight(SerialWeighted layer)
        {
            float std = MathF.Sqrt(2f / (layer.FanIn + layer.FanOut));

            return Utility.RandomGauss(0, std);
        }
    }
}
