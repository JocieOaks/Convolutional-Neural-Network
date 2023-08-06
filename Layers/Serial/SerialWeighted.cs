using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public abstract class SerialWeighted : ISerial
    {
        [JsonProperty] protected Weights _weights;
        [JsonProperty] protected Weights _bias;

        public SerialWeighted(Weights weights, Weights bias)
        {
            _weights = weights;
            _bias = bias;
        }

        [JsonConstructor] protected SerialWeighted() { }

        public abstract Shape Initialize(Shape inputShape);

        protected void InitializeWeights()
        {
            _weights.Initialize(WeightLength, this);
            _bias?.Initialize(BiasLength, this);
        }

        public void GetWeights(List<Weights> weights)
        {
            weights.Add(_weights);
            if(_bias != null)
            {
                weights.Add(_bias);
            }
        }

        public abstract Layer Construct();

        protected int WeightLength { get; set; }

        protected int BiasLength { get; set; }

        public int FanIn { get; protected set; }

        public int FanOut { get; protected set; }
    }
}
