using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers
{
    public class InputLayer : Layer, IReflexiveLayer
    {
        public override string Name => "Input Layer";

        private Tensor[] _input;

        public InputLayer(TensorShape inputShape)
        {
            _inputShape = inputShape;
        }

        public override void Backwards(int batchSize, bool update)
        {
        }

        public override void Forward(int batchSize)
        {
            for(int i = 0; i < batchSize; i++)
            {
                _input[i].CopyToView(_buffers.Input.SubView(i * _inputShape.Volume, _inputShape.Volume));
            }
        }

        public void SetInput(Tensor[] input)
        {
            _input = input;
        }

        public override TensorShape Startup(TensorShape inputShape, PairedBuffers buffers, int maxBatchSize)
        {
            _outputShape = _inputShape;
            _buffers = buffers;
            _buffers.OutputDimensionArea(_inputShape.Volume);
            return _inputShape;
        }
    }
}
