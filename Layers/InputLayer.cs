using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers
{
    public class InputLayer : Layer, IReflexiveLayer
    {
        public override string Name => "Input Layer";

        private Tensor[][] _input;

        public InputLayer(Shape inputShape)
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
                for(int j = 0; j < _inputShape.Dimensions; j++)
                {
                    if (_input[i][j].Area != _inputShape.Area)
                    {
                        throw new ArgumentException("Input images are incorrectly sized.");
                    }

                    _input[i][j].CopyToView(_buffers.Input.SubView(_inputShape.Area * (i * _inputShape.Dimensions + j), _inputShape.Area));
                }
            }
        }

        public void SetInput(Tensor[][] input)
        {
            _input = input;
        }

        public override Shape Startup(Shape inputShape, PairedBuffers buffers, int maxBatchSize)
        {
            _outputShape = _inputShape;
            _buffers = buffers;
            _buffers.OutputDimensionArea(_inputShape.Volume);
            return _inputShape;
        }
    }
}
