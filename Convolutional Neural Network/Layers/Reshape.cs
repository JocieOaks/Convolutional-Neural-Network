using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="Reshape"/> class is a <see cref="Layer"/> for changing the <see cref="TensorShape"/> from the input and the output.
    /// Note: The <see cref="Reshape"/> layer itself does nothing during training.
    /// It is there to change how the following <see cref="Layer"/> interprets the input it is given.
    /// </summary>
    public class Reshape : Layer
    { 
        /// <summary>
        /// Initializes a new instance of the <see cref="Reshape"/> class.
        /// </summary>
        /// <param name="outputShape">The <see cref="TensorShape"/> of the output.</param>
        public Reshape(TensorShape outputShape)
        {
            OutputShape = outputShape;
        }

        /// <inheritdoc />
        public override string Name => "Reshape Layer";

        /// <inheritdoc />
        public override bool Reflexive => true;

        /// <inheritdoc />
        public override void Backwards(int batchSize, bool update)
        {
        }

        /// <inheritdoc />
        public override void Forward(int batchSize)
        {
        }
        /// <inheritdoc />
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int maxBatchSize)
        {
            if (Initialized)
                return OutputShape;
            Initialized = true;

            int inputLength = inputShape.Volume;

            int outputLength = OutputShape.Volume;

            if(inputLength != outputLength)
            {
                throw new ArgumentException("Cannot reshape input into output shape. Input and output shapes have different lengths.");
            }

            return OutputShape;
        }
    }
}
