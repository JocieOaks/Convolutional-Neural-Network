using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.Initializers;
using ConvolutionalNeuralNetwork.Layers.Weighted;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialBatchNorm : SerialWeighted
    {
        public SerialBatchNorm()
        {
            _bias ??= new Weights(new Constant(0));
            _weights ??= new Weights(new Constant(1))
            {
                IgnoreClip = true
            };
        }

        public override Layer Construct()
        {
            return new BatchNormalization(_weights, _bias);
        }

        public override Shape Initialize(Shape inputShape)
        {
            WeightLength = inputShape.Dimensions;
            BiasLength = inputShape.Dimensions;
            InitializeWeights();
            return inputShape;
        }

        
    }
}
