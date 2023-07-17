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
        protected uint _batchSize;

        protected IOBuffers _buffers;
        [JsonProperty] protected int _filterSize;
        protected int _inputDimensions;
        protected ILayerInfo[] _layerInfos;
        protected int _outputDimensions;
        protected Shape[] _inputShapes;
        protected Shape[] _outputShapes;
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
        public abstract Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, uint batchSize);

        /// <summary>
        /// Initializes the <see cref="Layer"/> and many of its fields.
        /// </summary>
        /// <param name="inputs">The previous <see cref="Layer"/>'s output.</param>
        /// <param name="outGradients">The previous <see cref="Layer"/>'s inGradient.</param>
        /// <param name="outputDimensionFactor">A factor relating the number of input layers to the number of output layers.
        /// A positive number multiplies the number of input dimensions. A negative number divides the number of dimensions.</param>
        /// <exception cref="ArgumentException">Thrown if the ratio of input layers and output layers is not an integer.</exception>
        protected void BaseStartup(Shape[] inputShapes, IOBuffers buffers, uint batchSize, int outputDimensionFactor = 1)
        {
            _inputDimensions = inputShapes.Length;
            if (outputDimensionFactor >= 1)
            {
                _outputDimensions = outputDimensionFactor * _inputDimensions;
            }
            else
            {
                if (outputDimensionFactor == 0 || _inputDimensions % outputDimensionFactor != 0)
                {
                    throw new ArgumentException("outputDimensionFactor does not divide evenly with input dimensions.");
                }
                else
                {
                    _outputDimensions = _inputDimensions / -outputDimensionFactor;
                }
            }

            _batchSize = batchSize;
            _layerInfos = new ILayerInfo[_inputDimensions];
            _inputShapes = inputShapes;
            _outputShapes = new Shape[_outputDimensions];

            for (int i = 0; i < _inputDimensions; i++)
            {
                if (_stride == 1 && _filterSize == 1)
                {
                    _layerInfos[i] = new StaticLayerInfo()
                    {
                        Width = inputShapes[i].Width,
                        Length = inputShapes[i].Length,
                    };
                }
                else
                {

                    int outputWidth = (int)MathF.Ceiling(inputShapes[i].Width / (float)_stride);
                    int outputLength = (int)MathF.Ceiling(inputShapes[i].Length / (float)_stride);
                    _layerInfos[i] = new LayerInfo()
                    {
                        FilterSize = _filterSize,
                        Stride = _stride,
                        InverseKSquared = 1f / (_filterSize * _filterSize),
                        InputWidth = inputShapes[i].Width,
                        InputLength = inputShapes[i].Length,
                        OutputWidth = outputWidth,
                        OutputLength = outputLength,
                        Padding = _filterSize - _stride 
                    };
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                ILayerInfo layer;
                if (outputDimensionFactor >= 1)
                {
                    layer = _layerInfos[i / outputDimensionFactor];
                }
                else
                {
                    layer = _layerInfos[i * -outputDimensionFactor];
                }

                for (int j = 0; j < _batchSize; j++)
                {
                    _outputShapes[i] = new Shape(layer.OutputWidth, layer.OutputLength);
                }
            }

            _buffers = buffers;
            for (int i = 0; i < _outputDimensions; i++)
                buffers.OutputDimensionArea(i, _outputShapes[i].Area);
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

        protected FeatureMap[,] FilterTestSetup(int dimensionMultiplier)
        {
            int outputDimensions, inputDimensions;
            if (dimensionMultiplier >= 1)
            {
                inputDimensions = 1;
                outputDimensions = dimensionMultiplier;
            }
            else
            {
                inputDimensions = -dimensionMultiplier;
                outputDimensions = 1;
            }

            FeatureMap[,] inputs = new FeatureMap[inputDimensions, 1];
            Shape[] inputShape = new Shape[inputDimensions];
            for (int i = 0; i < inputDimensions; i++)
            {
                inputs[i, 0] = new(3, 3);
                inputShape[i] = new Shape(3, 3);
            }

            IOBuffers buffer = new();
            IOBuffers complimentBuffer = new();
            complimentBuffer.OutputDimensionArea(inputDimensions - 1, 9);

            Startup(inputShape, buffer, 1);
            buffer.Allocate(1);
            complimentBuffer.Allocate(1);
            IOBuffers.SetCompliment(buffer, complimentBuffer);

            return inputs;
        }
    }
}