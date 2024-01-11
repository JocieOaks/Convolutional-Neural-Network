using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    public class Input : Layer
    {
        public override string Name => "Input Layer";

        private Tensor[] _input;

        public Input(TensorShape inputShape)
        {
            InputShape = inputShape;
        }

        public override void Backwards(int batchSize, bool update)
        {
        }

        public override void Forward(int batchSize)
        {
            for(int i = 0; i < batchSize; i++)
            {
                _input[i].CopyToView(Buffers.Input.SubView(i * InputShape.Volume, InputShape.Volume));
            }
        }

        /// <inheritdoc />
        [JsonIgnore] public override bool Reflexive => true;

        public void SetInput(Tensor[] input)
        {
            _input = input;
        }

        public override TensorShape Startup(TensorShape inputShape, PairedBuffers buffers, int maxBatchSize)
        {
            OutputShape = InputShape;
            Buffers = buffers;
            Buffers.OutputDimensionArea(InputShape.Volume);
            return InputShape;
        }
    }
}
