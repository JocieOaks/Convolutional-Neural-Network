using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="Labels"/> class is a <see cref="Layer"/> for important an image's labels into the <see cref="Network"/>.
    /// </summary>
    public class Labels : Layer
    {
        private readonly int _numLabels;
        private Vector[] _input;

        /// <summary>
        /// Initializes a new instance of the <see cref="Input"/> class.
        /// </summary>
        /// <param name="labelCount">Number of classifying labels.</param>
        public Labels(int labelCount)
        {
            InputShape = new TensorShape(labelCount, 1, 1);
            _numLabels = labelCount;
        }

        /// <inheritdoc />
        public override string Name => "Labels Layer";

        /// <inheritdoc />
        public override bool Reflexive => true;

        /// <inheritdoc />
        public override void Backwards(int batchSize, bool update)
        {
        }

        /// <inheritdoc />
        public override void Forward(int batchSize)
        {
            for (int i = 0; i < batchSize; i++)
            {
                _input[i].CopyToView(Views.Input.SubView(i * InputShape.Volume, InputShape.Volume));
            }
        }

        /// <summary>
        /// Set the input to the <see cref="Network"/> to be copied to the GPU.
        /// </summary>
        public void SetLabels(Vector[] labels)
        {
            if (labels[0].Length != _numLabels)
            {
                throw new ArgumentException("Incorrect number of labels.");
            }
            _input = labels;
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
