using ConvolutionalNeuralNetwork.Layers.Serial.Weighted;

namespace ConvolutionalNeuralNetwork.DataTypes.Initializers
{
    /// <summary>
    /// The <see cref="IWeightInitializer"/> interface is used by <see cref="Weights"/> to set the initial values for a <see cref="SerialWeighted"/> layer.
    /// </summary>
    public interface IWeightInitializer
    {
        /// <summary>
        /// Returns a single value to be used by <see cref="Weights"/> based on the specific details of the given <see cref="SerialWeighted"/>,
        /// such as the fan in/out and length.
        /// </summary>
        float GetWeight(SerialWeighted layer);
    }
}
