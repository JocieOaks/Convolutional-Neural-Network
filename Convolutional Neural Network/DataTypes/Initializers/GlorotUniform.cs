using ConvolutionalNeuralNetwork.Layers.Serial.Weighted;

namespace ConvolutionalNeuralNetwork.DataTypes.Initializers
{
    /// <summary>
    /// The <see cref="GlorotNormal"/> class is an <see cref="IWeightInitializer"/> that returns values in a uniform distribution,
    /// based on the fan in/out of the <see cref="SerialWeighted"/> layer.
    /// </summary>
    public class GlorotUniform : IWeightInitializer
    {
        /// <value>A static instance of the <see cref="GlorotUniform"/> class.</value>
        public static GlorotUniform Instance { get; } = new();

        /// <inheritdoc />
        public float GetWeight(SerialWeighted layer)
        {
            float limit = MathF.Sqrt(6f / (layer.FanIn + layer.FanOut));

            return (Utility.Random.NextSingle() * 2 - 1) * limit;
        }
    }
}
