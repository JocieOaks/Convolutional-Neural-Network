using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.Weighted;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial.Weighted
{
    /// <summary>
    /// The <see cref="SerialDense"/> class is a <see cref="SerialWeighted"/> for <see cref="Dense"/> layers.
    /// </summary>
    public class SerialDense : SerialWeighted
    {
        [JsonProperty] private int _outputUnits;

        /// <summary>
        /// Initializes a new instance of the <see cref="SerialDense"/> class.
        /// </summary>
        /// <param name="outputUnits">The length of the <see cref="Layer"/>'s output <see cref="Vector"/>.</param>
        /// <param name="weights">The <see cref="Weights"/> used by the <see cref="WeightedLayer"/>.</param>
        /// <param name="bias">The <see cref="Weights"/> used for bias.</param>
        public SerialDense(int outputUnits, Weights weights, Weights bias) : base(weights, bias)
        {
            _outputUnits = outputUnits;
        }

        [JsonConstructor] private SerialDense() { }

        /// <inheritdoc />
        public override Layer Construct()
        {
            return new Dense(_outputUnits, Weights, Bias);
        }

        /// <inheritdoc />
        public override TensorShape Initialize(TensorShape inputShape)
        {
            TensorShape outputShape = new(_outputUnits, 1, 1);

            FanIn = inputShape.Volume;
            FanOut = _outputUnits;
            WeightLength = inputShape.Volume * _outputUnits;
            BiasLength = 1;
            InitializeWeights();

            return outputShape;
        }
    }
}
