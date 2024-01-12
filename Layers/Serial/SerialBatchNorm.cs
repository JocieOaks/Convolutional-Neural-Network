using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.Weighted;
using ConvolutionalNeuralNetwork.DataTypes.Initializers;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialBatchNorm : SerialWeighted
    {
        public SerialBatchNorm()
        {
            _bias ??= new Weights(new Constant(0));
            _weights ??= new Weights(new Constant(1));
        }

        public override Layer Construct()
        {
            return new BatchNormalization(_weights, _bias);
        }

        public override TensorShape Initialize(TensorShape inputShape)
        {
            WeightLength = inputShape.Dimensions;
            BiasLength = inputShape.Dimensions;
            InitializeWeights();
            return inputShape;
        }

        
    }
}
