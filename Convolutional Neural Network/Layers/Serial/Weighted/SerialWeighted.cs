using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.Weighted;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial.Weighted
{
    /// <summary>
    /// The <see cref="SerialWeighted"/> class is an abstract <see cref="ISerialLayer"/> for <see cref="WeightedLayer"/>s.
    /// </summary>
    public abstract class SerialWeighted : ISerialLayer
    {
        /// <value>The <see cref="DataTypes.Weights"/> used for bias.</value>
        [JsonProperty] protected Weights Bias;

        /// <value>The <see cref="DataTypes.Weights"/> used by the <see cref="WeightedLayer"/>.</value>
        [JsonProperty] protected Weights Weights;
        /// <summary>
        /// Initializes a new instance of the <see cref="SerialWeighted"/> class.
        /// </summary>
        /// <param name="weights">The <see cref="DataTypes.Weights"/> used by the <see cref="WeightedLayer"/>.</param>
        /// <param name="bias">The <see cref="DataTypes.Weights"/> used for bias.</param>
        protected SerialWeighted(Weights weights, Weights bias)
        {
            Weights = weights;
            Bias = bias;
        }

        /// <summary>
        /// A json constructor used for deserializing.
        /// </summary>
        [JsonConstructor] protected SerialWeighted() { }

        /// <value>The fan in for the <see cref="Layer"/>.</value>
        public int FanIn { get; protected set; }

        /// <value>The fan out for the <see cref="Layer"/>.</value>
        public int FanOut { get; protected set; }

        /// <value>The length of <see cref="Bias"/>.</value>
        protected int BiasLength { get; set; }

        /// <value>The length of <see cref="Weights"/>.</value>
        protected int WeightLength { get; set; }

        /// <inheritdoc />
        public abstract Layer Construct();

        /// <summary>
        /// Adds the <see cref="SerialWeighted"/>'s <see cref="DataTypes.Weights"/> to the given list of <see cref="DataTypes.Weights"/>.
        /// </summary>
        public void GetWeights(List<Weights> weights)
        {
            weights.Add(Weights);
            if (Bias != null)
            {
                weights.Add(Bias);
            }
        }

        /// <inheritdoc />
        public abstract TensorShape Initialize(TensorShape inputShape);

        /// <summary>
        /// Initializes the <see cref="Weights"/> and <see cref="Bias"/> of the <see cref="SerialWeighted"/>.
        /// </summary>
        protected void InitializeWeights()
        {
            Weights.InitializeWeights(WeightLength, this);
            Bias?.InitializeWeights(BiasLength, this);
        }
    }
}
