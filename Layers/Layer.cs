using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="Layer"/> is an abstract base class for all the layer's in a <see cref="Network"/>.
    /// </summary>
    [Serializable]
    public abstract class Layer
    {
        [JsonIgnore] protected PairedBuffers Buffers { get; set; }
        protected int FilterSize { get; set; }
        [JsonIgnore] protected TensorShape InputShape { get; set; }
        [JsonIgnore] protected LayerInfo LayerInfo { get; set; }
        [JsonIgnore] protected TensorShape OutputShape { get; set; }
        [JsonIgnore] protected bool Ready { get; set; } = false;
        protected int Stride { get; set; }
        /// <summary>
        /// Initializes a new instance of the <see cref="Layer"/> class.
        /// </summary>
        /// <param name="filterSize">The width and height of a filter.</param>
        /// <param name="stride">The amount of movement over the image for each filter pass.</param>
        protected Layer(int filterSize, int stride)
        {
            FilterSize = filterSize;
            Stride = stride;
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor]
        protected Layer()
        {
        }

        /// <value>The <see cref="ArrayView{T}"/> of the input of the <see cref="Layer"/>.</value>
        [JsonIgnore] public ArrayView<float> Input => Buffers.Input;
        /// <value>The name of the <see cref="Layer"/>, used for logging.</value>
        [JsonIgnore] public abstract string Name { get; }

        /// <value>The <see cref="ArrayView{T}"/> of the output of the <see cref="Layer"/>.</value>
        [JsonIgnore] public ArrayView<float> Output => Reflexive ? Buffers.Input : Buffers.Output;

        /// <value>Indicates whether the layer is reflexive. Reflexive layers only modify the input view, instead of modifying the output view.</value>
        [JsonIgnore] public virtual bool Reflexive => false;

        /// <summary>
        /// Back-propagates through the <see cref="Layer"/> updating any layer weights, and calculating the outgoing gradient that is
        /// shared with the previous layer.
        /// </summary>
        /// <param name="batchSize">The size of the current batch of training images.</param>
        /// <param name="update">Whether weights should be updated during back-propagation.</param>
        public abstract void Backwards(int batchSize, bool update);

        /// <summary>
        /// Forward propagates through the <see cref="Layer"/> calculating the output <see cref="FeatureMap"/> that is shared with
        /// the next layer.
        /// </summary>
        /// <param name="batchSize">The size of the current batch of training images.</param>
        public abstract void Forward(int batchSize);

        /// <summary>
        /// Initializes the <see cref="Layer"/> for the data set being used.
        /// </summary>
        /// <param name="inputShape">The <see cref="TensorShape"/> of the previous <see cref="Layer"/>'s output.</param>
        /// <param name="buffers">The <see cref="PairedBuffers"/> containing the input and output buffers.</param>
        /// <param name="maxBatchSize">The maximum size of training batches.</param>
        /// <returns>Returns the output and inGradient to share with the next <see cref="Layer"/>.</returns>
        public abstract TensorShape Startup(TensorShape inputShape, PairedBuffers buffers, int maxBatchSize);

        protected static void Synchronize()
        {
            GPUManager.Accelerator.Synchronize();
        }

        /// <summary>
        /// Initializes the <see cref="Layer"/> and many of its fields.
        /// </summary>
        /// <param name="inputShape">The <see cref="TensorShape"/> of the previous <see cref="Layer"/>'s output.</param>
        /// <param name="buffers">The <see cref="PairedBuffers"/> containing the input and output buffers.</param>
        /// <param name="outputDimensions">A factor relating the number of input layers to the number of output layers.
        /// A positive number multiplies the number of input dimensions. A negative number divides the number of dimensions.</param>
        /// <exception cref="ArgumentException">Thrown if the ratio of input layers and output layers is not an integer.</exception>
        protected void BaseStartup(TensorShape inputShape, PairedBuffers buffers, int outputDimensions = -1)
        {
            InputShape = inputShape;
            if(outputDimensions == -1)
            { 
                outputDimensions = InputShape.Dimensions;
            }

            if (Stride == 1 && FilterSize == 1 && Name != "Convolutional Layer")
            {
                OutputShape = new TensorShape(inputShape.Width, inputShape.Length, outputDimensions);
            }
            else
            {
                int outputWidth = (int)MathF.Ceiling(inputShape.Width / (float)Stride);
                int outputLength = (int)MathF.Ceiling(inputShape.Length / (float)Stride);
                OutputShape = new TensorShape(outputWidth, outputLength, outputDimensions);
                LayerInfo = new LayerInfo(inputShape, OutputShape, FilterSize, Stride);
            }

            Buffers = buffers;
            buffers.OutputDimensionArea(OutputShape.Volume);
        }
    }
}