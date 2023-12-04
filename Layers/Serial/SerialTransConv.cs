using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.Weighted;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialTransConv : SerialWeighted
    {
        [JsonProperty] private readonly int _filterSize;
        [JsonProperty] private readonly int _stride;
        [JsonProperty] private readonly int _outputDimensions;
        [JsonProperty] private int _inputDimensions;


        public SerialTransConv(int dimensions, int filterSize, int stride, Weights weights, Weights bias) : base(weights, bias)
        {
            _outputDimensions = dimensions;
            _filterSize = filterSize;
            _stride = stride;
        }

        [JsonConstructor] private SerialTransConv() { }

        public override Layer Construct()
        {
            return new TransposeConvolution(_filterSize, _stride, _outputDimensions, _weights, _bias);
        }

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

            TensorShape outputShape = new TensorShape(inputShape.Width * _stride, inputShape.Length * _stride, _outputDimensions);

            FanIn = inputShape.Volume;
            FanOut = outputShape.Volume;
            WeightLength = _filterSize * _filterSize * _outputDimensions * _inputDimensions;
            BiasLength = _outputDimensions;
            InitializeWeights();

            return outputShape;
        }
    }
}
