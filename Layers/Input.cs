using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="Input"/> class is a <see cref="Layer"/> for copying the input to the <see cref="Network"/> into the GPU views.
    /// </summary>
    public class Input : Layer
    {
        private Tensor[] _input;

        /// <summary>
        /// Initializes a new instance of the <see cref="Input"/> class.
        /// </summary>
        /// <param name="inputShape">Shape of the <see cref="Network"/>'s input.</param>
        public Input(TensorShape inputShape)
        {
            InputShape = inputShape;
        }

        /// <inheritdoc />
        public override string Name => "Input Layer";

        /// <inheritdoc />
        public override bool Reflexive => true;

        /// <inheritdoc />
        public override void Backwards(int batchSize, bool update)
        {
        }

        /// <inheritdoc />
        public override void Forward(int batchSize)
        {
            for(int i = 0; i < batchSize; i++)
            {
                _input[i].CopyToView(Views.Input.SubView(i * InputShape.Volume, InputShape.Volume));
            }
        }

        /// <summary>
        /// Set the input to the <see cref="Network"/> to be copied to the GPU.
        /// </summary>
        public void SetInput(Tensor[] input)
        {
            _input = input;
        }

        /// <inheritdoc />
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int maxBatchSize)
        {
            OutputShape = InputShape;
            Views = views;
            views.OutputDimensionArea(InputShape.Volume);
            return InputShape;
        }
    }
}
