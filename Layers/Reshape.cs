using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    public class Reshape : Layer, IStructuralLayer, IUnchangedLayer
    { 
        public Reshape(Shape outputShape)
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

        public override Shape Startup(Shape inputShape, IOBuffers buffers, int maxBatchSize)
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
