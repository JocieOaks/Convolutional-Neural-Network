using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers.Loss
{
    /// <summary>
    /// The <see cref="Loss"/> class takes an output <see cref="Tensor"/> from a series of <see cref="Layer"/>s
    /// and determines the loss when compared to the ground truth, and then creates a gradient <see cref="Tensor"/>
    /// to be used for back-propagating through the <see cref="Layer"/>s.
    /// </summary>
    public abstract class Loss
    {
        /// <summary>
        /// The <see cref="PairedGPUViews"/> containing the output of the <see cref="Network"/>,
        /// and used to store the gradients for back-propagation.
        /// </summary>
        protected PairedGPUViews Views;

        /// <summary>
        /// The <see cref="TensorShape"/> for the output of the <see cref="Network"/>.
        /// </summary>
        protected TensorShape OutputShape;

        /// <summary>
        /// A <see cref="Vector"/> used to store the labels for determining loss calculations.
        /// </summary>
        protected Vector Labels;

        /// <summary>
        /// A <see cref="Vector"/> used to store the classification of an input, and whether it is fake or real.
        /// </summary>
        protected Vector Classifications;

        /// <summary>
        /// A single dimensional <see cref="Vector"/> for storing the total loss of the <see cref="Network"/>.
        /// </summary>
        protected readonly Vector Losses = new(1);

        /// <summary>
        /// A single dimension <see cref="Vector"/> for storing the total accuracy of the <see cref="Network"/>.
        /// </summary>
        protected readonly Vector Accuracy = new(1);

        /// <summary>
        /// Initializes the <see cref="Loss"/> for the data set being used.
        /// </summary>
        /// <param name="views">The <see cref="PairedGPUViews"/> containing the output of the <see cref="Network"/>,
        /// and used to store the gradients for back-propagation.</param>
        /// <param name="outputShape">The <see cref="TensorShape"/> for the output of the <see cref="Network"/>.</param>
        /// <param name="maxBatchSize">The maximum size of training batches.</param>
        public virtual void Startup(PairedGPUViews views, TensorShape outputShape, int maxBatchSize)
        {
            Views = views;
            OutputShape = outputShape;
            Labels = new Vector(maxBatchSize * outputShape.Volume);
            Classifications = new Vector(maxBatchSize);
        }

        /// <summary>
        /// Calculates the loss and accuracy of the <see cref="Network"/>, and then creates the gradients
        /// for back-propagating through the <see cref="Network"/>.
        /// </summary>
        /// <param name="labels"></param>
        /// <param name="classifications"></param>
        /// <returns>Returns the loss and accuracy as a tuple of floats.</returns>
        public abstract (float, float) GetLoss(Vector[] labels, Vector classifications);
    }
}
