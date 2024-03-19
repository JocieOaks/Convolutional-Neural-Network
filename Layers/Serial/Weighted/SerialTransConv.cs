using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.Weighted;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial.Weighted
{
    /// <summary>
    /// The <see cref="SerialTransConv"/> class is a <see cref="SerialWeighted"/> for <see cref="TransposeConvolution"/> layers.
    /// </summary>
    public class SerialTransConv : SerialWeighted
    {
        [JsonProperty] private readonly int _filterSize;
        [JsonProperty] private readonly int _stride;
        [JsonProperty] private readonly int _outputDimensions;
        [JsonProperty] private int _inputDimensions;

        /// <summary>
        /// Initializes a new instance of the <see cref="SerialTransConv"/> class.
        /// </summary>
        /// <param name="dimensions">The dimensions of the <see cref="Layer"/>'s output.</param>
        /// <param name="filterSize">The width and length of the <see cref="Layer"/>'s filters.</param>
        /// <param name="stride">The <see cref="Layer"/>'s horizontal and vertical stride.</param>
        /// <param name="weights">The <see cref="Weights"/> used by the <see cref="WeightedLayer"/>.</param>
        /// <param name="bias">The <see cref="Weights"/> used for bias.</param>
        public SerialTransConv(int dimensions, int filterSize, int stride, Weights weights, Weights bias) : base(weights, bias)
        {
            _outputDimensions = dimensions;
            _filterSize = filterSize;
            _stride = stride;
        }

        [JsonConstructor] private SerialTransConv() { }

        /// <inheritdoc />
        public override Layer Construct()
        {
            return new TransposeConvolution(_filterSize, _stride, _outputDimensions, Weights, Bias);
        }

        /// <inheritdoc />
        public override TensorShape Initialize(TensorShape inputShape)
        {
            if (_inputDimensions != 0)
            {
                if (inputShape.Dimensions != _inputDimensions)
                {
                    throw new ArgumentException("Convolution layer is incompatible with input shape.");
                }
            }
            else
            {
                _inputDimensions = inputShape.Dimensions;
            }

            TensorShape outputShape = new(inputShape.Width * _stride, inputShape.Length * _stride, _outputDimensions);

            FanIn = inputShape.Volume;
            FanOut = outputShape.Volume;
            WeightLength = _filterSize * _filterSize * _outputDimensions * _inputDimensions;
            BiasLength = _outputDimensions;
            InitializeWeights();

            return outputShape;
        }
    }
}
