using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.Weighted;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialConv : SerialWeighted
    {
        [JsonProperty] private readonly int _filterSize;
        [JsonProperty] private readonly int _stride;
        [JsonProperty] private readonly int _outputDimensions;
        [JsonProperty] private int _inputDimensions;


        public SerialConv(int dimensions, int filterSize, int stride, Weights weights, Weights bias) : base(weights, bias)
        {
            _outputDimensions = dimensions;
            _filterSize = filterSize;
            _stride = stride;
        }

        [JsonConstructor] private SerialConv() { }

        public override Layer Construct()
        {
            return new Convolution(_filterSize, _stride, _outputDimensions, _weights, _bias);
        }

        public override TensorShape Initialize(TensorShape inputShape)
        {
            if(_inputDimensions != 0)
            {
                if(inputShape.Dimensions != _inputDimensions)
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
