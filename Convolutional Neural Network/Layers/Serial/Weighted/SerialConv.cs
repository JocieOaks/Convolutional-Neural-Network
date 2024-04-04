using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.Weighted;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial.Weighted
{
    /// <summary>
    /// The <see cref="SerialConv"/> class is a <see cref="SerialWeighted"/> for <see cref="Convolution"/> layers.
    /// </summary>
    public class SerialConv : SerialWeighted
    {
        [JsonProperty] private readonly int _filterSize;
        [JsonProperty] private readonly int _stride;
        [JsonProperty] private readonly int _outputDimensions;
        [JsonProperty] private int _inputDimensions;

        /// <summary>
        /// Initializes a new instance of the <see cref="SerialConv"/> class.
        /// </summary>
        /// <param name="dimensions">The dimensions of the <see cref="Layer"/>'s output.</param>
        /// <param name="filterSize">The width and length of the <see cref="Layer"/>'s filters.</param>
        /// <param name="stride">The <see cref="Layer"/>'s horizontal and vertical stride.</param>
        /// <param name="weights">The <see cref="Weights"/> used by the <see cref="WeightedLayer"/>.</param>
        /// <param name="bias">The <see cref="Weights"/> used for bias.</param>
        public SerialConv(int dimensions, int filterSize, int stride, Weights weights, Weights bias) : base(weights, bias)
        {
            _outputDimensions = dimensions;
            _filterSize = filterSize;
            _stride = stride;
        }

        [JsonConstructor] private SerialConv() { }

        /// <inheritdoc />
        public override Layer Construct()
        {
            return new Convolution(_filterSize, _stride, _outputDimensions, Weights, Bias);
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

            int outputWidth = (int)MathF.Ceiling(inputShape.Width / (float)_stride);
            int outputLength = (int)MathF.Ceiling(inputShape.Length / (float)_stride);
            TensorShape outputShape = new(outputWidth, outputLength, _outputDimensions);

            FanIn = inputShape.Volume;
            FanOut = outputShape.Volume;
            WeightLength = _filterSize * _filterSize * _outputDimensions * _inputDimensions;
            BiasLength = _outputDimensions;
            InitializeWeights();

            return outputShape;
        }
    }
}
