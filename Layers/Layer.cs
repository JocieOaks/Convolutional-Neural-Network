using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using Newtonsoft.Json;
using System.Reflection.Emit;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="Layer"/> is an abstract base class for all the layer's in a <see cref="Network"/>.
    /// </summary>
    [Serializable]
    public abstract class Layer : ILayer
    {
        protected IOBuffers _buffers;
        [JsonProperty] protected int _filterSize;
        protected Shape _inputShape;
        protected ILayerInfo _layerInfo;
        protected Shape _outputShape;
        [JsonProperty] protected int _stride;

        protected bool _ready = false;

        [JsonIgnore] public Shape OutputShape => _outputShape;

        /// <summary>
        /// Initializes a new instance of the <see cref="Layer"/> class.
        /// </summary>
        /// <param name="filterSize">The width and height of a filter.</param>
        /// <param name="stride">The amount of movement over the image for each filter pass.</param>
        public Layer(int filterSize, int stride)
        {
            _filterSize = filterSize;
            _stride = stride;
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor]
        protected Layer()
        {
        }

        [JsonIgnore] public ArrayView<float> InGradient => this is IUnchangedLayer ? _buffers.OutGradient : _buffers.InGradient;
        [JsonIgnore] public ArrayView<float> Input => _buffers.Input;
        /// <inheritdoc/>
        [JsonIgnore] public abstract string Name { get; }

        [JsonIgnore] public ArrayView<float> OutGradient => _buffers.OutGradient;
        [JsonIgnore] public ArrayView<float> Output => this is IUnchangedLayer ? _buffers.Input : _buffers.Output;
        /// <inheritdoc/>
        public abstract void Backwards(int batchSize);

        /// <inheritdoc/>
        public abstract void Forward(int batchSize);

        /// <inheritdoc/>
        public abstract void Reset();

        /// <inheritdoc/>
        public abstract Shape Startup(Shape inputShape, IOBuffers buffers, int maxBatchSize);

        protected static void Synchronize()
        {
            GPUManager.Accelerator.Synchronize();
        }

        /// <summary>
        /// Initializes the <see cref="Layer"/> and many of its fields.
        /// </summary>
        /// <param name="inputs">The previous <see cref="Layer"/>'s output.</param>
        /// <param name="outGradients">The previous <see cref="Layer"/>'s inGradient.</param>
        /// <param name="outputDimensions">A factor relating the number of input layers to the number of output layers.
        /// A positive number multiplies the number of input dimensions. A negative number divides the number of dimensions.</param>
        /// <exception cref="ArgumentException">Thrown if the ratio of input layers and output layers is not an integer.</exception>
        protected void BaseStartup(Shape inputShape, IOBuffers buffers, int outputDimensions = -1)
        {
            _inputShape = inputShape;
            if(outputDimensions == -1)
            { 
                outputDimensions = _inputShape.Dimensions;
            }

            if (_stride == 1 && _filterSize == 1 && Name != "Convolutional Layer")
            {
                _outputShape = new Shape(inputShape.Width, inputShape.Length, outputDimensions);
            }
            else
            {
                int outputWidth = (int)MathF.Ceiling(inputShape.Width / (float)_stride);
                int outputLength = (int)MathF.Ceiling(inputShape.Length / (float)_stride);
                _outputShape = new Shape(outputWidth, outputLength, outputDimensions);
                _layerInfo = new LayerInfo(inputShape, _outputShape, _filterSize, _stride);
            }

            _buffers = buffers;
            buffers.OutputDimensionArea(_outputShape.Volume);
        }
    }
}