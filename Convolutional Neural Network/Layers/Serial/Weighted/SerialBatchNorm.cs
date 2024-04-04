using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.DataTypes.Initializers;
using ConvolutionalNeuralNetwork.Layers.Weighted;

namespace ConvolutionalNeuralNetwork.Layers.Serial.Weighted
{
    /// <summary>
    /// The <see cref="SerialBatchNorm"/> class is a <see cref="SerialWeighted"/> for <see cref="BatchNormalization"/> layers.
    /// </summary>
    public class SerialBatchNorm : SerialWeighted
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BatchNormalization"/> class.
        /// </summary>
        public SerialBatchNorm()
        {
            Bias ??= new Weights(new Constant(0));
            Weights ??= new Weights(new Constant(1));
        }

        /// <inheritdoc />
        public override Layer Construct()
        {
            return new BatchNormalization(Weights, Bias);
        }

        /// <inheritdoc />
        public override TensorShape Initialize(TensorShape inputShape)
        {
            WeightLength = inputShape.Dimensions;
            BiasLength = inputShape.Dimensions;
            InitializeWeights();
            return inputShape;
        }


    }
}
