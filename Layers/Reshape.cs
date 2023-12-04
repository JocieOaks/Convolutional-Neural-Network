using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers
{
    public class Reshape : Layer, IReflexiveLayer
    { 
        public Reshape(TensorShape outputShape)
        {
            _outputShape = outputShape;
        }

        public override string Name => "Reshape Layer";

        public override void Backwards(int batchSize, bool update)
        {
        }

        public override void Forward(int batchSize)
        {
        }

        public override TensorShape Startup(TensorShape inputShape, PairedBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            int inputLength = inputShape.Volume;

            int outputLength = _outputShape.Volume;

            if(inputLength != outputLength)
            {
                throw new ArgumentException("Input and output shapes have different lengths.");
            }

            return _outputShape;
        }
    }
}
