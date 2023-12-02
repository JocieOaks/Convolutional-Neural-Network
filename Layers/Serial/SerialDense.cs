using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.Weighted;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialDense : SerialWeighted
    {
        [JsonProperty] private int _outputUnits;

        public SerialDense(int outputUnits, Weights weights, Weights bias) : base(weights, bias)
        {
            _outputUnits = outputUnits;
        }

        [JsonConstructor] private SerialDense() { }

        public override Layer Construct()
        {
            return new Dense(_outputUnits, _weights, _bias);
        }

        public override Shape Initialize(Shape inputShape)
        {
            Shape outputShape = new(_outputUnits, 1, 1);

            FanIn = inputShape.Volume;
            FanOut = _outputUnits;
            WeightLength = inputShape.Volume * _outputUnits;
            BiasLength = 1;
            InitializeWeights();

            return outputShape;
        }
    }
}
