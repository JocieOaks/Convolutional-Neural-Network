using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    /// <summary>
    /// The <see cref="SerialLabels"/> class is an <see cref="ISerialLayer"/> for <see cref="Labels"/> layers.
    /// </summary>
    public class SerialLabels : ISerialLayer
    {
        [JsonProperty] private int _numLabels;

        /// <summary>
        /// Initializes a new instance of the <see cref="SerialInput"/> class.
        /// </summary>
        /// <param name="labelCount">The numbers of classification labels.</param>
        public SerialLabels(int labelCount)
        {
            _numLabels = labelCount;
        }

        [JsonConstructor] private SerialLabels() { }

        /// <inheritdoc />
        public Layer Construct()
        {
            return new Labels(_numLabels);
        }

        /// <inheritdoc />
        public TensorShape Initialize(TensorShape inputShape)
        {
            return new TensorShape(_numLabels, 1, 1);
        }
    }
}