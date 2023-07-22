using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
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
        //All FeatureMaps are constant throughout the programs lifetime, and are updated with each pass through the network.
        //Arrays of FeatureMaps are shared between layers as references.
        protected int _batchSize;

        protected IOBuffers _buffers;
        [JsonProperty] protected int _filterSize;
        protected int _inputDimensions;
        protected ILayerInfo _layerInfo;
        protected int _outputDimensions;
        protected Shape _inputShape;
        protected Shape _outputShape;
        [JsonProperty] protected int _stride;

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

        /// <inheritdoc/>
        [JsonIgnore] public abstract string Name { get; }

        /// <inheritdoc/>
        public abstract void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay);

        /// <inheritdoc/>
        public abstract void Forward();

        /// <inheritdoc/>
        public abstract void Reset();

        /// <inheritdoc/>
        public abstract Shape Startup(Shape inputShapes, IOBuffers buffers, int batchSize);

        /// <summary>
        /// Initializes the <see cref="Layer"/> and many of its fields.
        /// </summary>
        /// <param name="inputs">The previous <see cref="Layer"/>'s output.</param>
        /// <param name="outGradients">The previous <see cref="Layer"/>'s inGradient.</param>
        /// <param name="outputDimensionFactor">A factor relating the number of input layers to the number of output layers.
        /// A positive number multiplies the number of input dimensions. A negative number divides the number of dimensions.</param>
        /// <exception cref="ArgumentException">Thrown if the ratio of input layers and output layers is not an integer.</exception>
        protected void BaseStartup(Shape inputShape, IOBuffers buffers, int batchSize, int outputDimensionFactor = 1)
        {
            _inputDimensions = inputShape.Dimensions;

            if(outputDimensionFactor != 1)
            {
                _outputDimensions = outputDimensionFactor;
            }
            else
            {
                _outputDimensions = _inputDimensions;
            }

            _batchSize = batchSize;
            _inputShape = inputShape;


            if (_stride == 1 && _filterSize == 1)
            {
                _layerInfo = new StaticLayerInfo()
                {
                    Width = inputShape.Width,
                    Length = inputShape.Length,
                    InputDimensions = _inputDimensions,
                    OutputDimensions = _outputDimensions
                };
                _outputShape = new Shape(inputShape.Width, inputShape.Length, _outputDimensions);
            }
            else
            {
                int outputWidth = (int)MathF.Ceiling(inputShape.Width / (float)_stride);
                int outputLength = (int)MathF.Ceiling(inputShape.Length / (float)_stride);
                _layerInfo = new LayerInfo()
                {
                    FilterSize = _filterSize,
                    Stride = _stride,
                    InverseKSquared = 1f / (_filterSize * _filterSize),
                    InputDimensions = _inputDimensions,
                    InputWidth = inputShape.Width,
                    InputLength = inputShape.Length,
                    OutputDimensions = _outputDimensions,
                    OutputWidth = outputWidth,
                    OutputLength = outputLength,
                    Padding = (_filterSize - 1) / 2
                };
                _outputShape = new Shape(outputWidth, outputLength, _outputDimensions);
            }

            _buffers = buffers;
            buffers.OutputDimensionArea(_outputDimensions * _outputShape.Area);
        }

        protected static void DecrementCacheabble(Cacheable[,] caches, uint decrement = 1)
        {
            for (int i = 0; i < caches.GetLength(0); i++)
            {
                for (int j = 0; j < caches.GetLength(1); j++)
                {
                    caches[i, j].DecrementLiveCount(decrement);
                }
            }
        }

        protected static void Synchronize()
        {
            GPUManager.Accelerator.Synchronize();
        }

        protected (Shape, Shape) FilterTestSetup(int inputDimensions, int batchSize, int inputSize)
        {
            Shape inputShape = new Shape(inputSize, inputSize, inputDimensions);
            

            IOBuffers buffer = new();
            IOBuffers complimentBuffer = new();
            complimentBuffer.OutputDimensionArea(inputDimensions * inputSize * inputSize);

            Shape outputShape = Startup(inputShape, buffer, batchSize);
            buffer.Allocate(batchSize);
            complimentBuffer.Allocate(batchSize);
            IOBuffers.SetCompliment(buffer, complimentBuffer);


            inputShape = new Shape(inputSize, inputSize, inputDimensions);

            return (inputShape, outputShape);
        }
    }
}