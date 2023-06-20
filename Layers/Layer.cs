﻿using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

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
        protected MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceInGradients;
        protected MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceInputs;
        protected MemoryBuffer1D<float, Stride1D.Dense>[,] _deviceOutGradients;
        protected MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceOutputs;
        [JsonProperty] protected int _filterSize;
        protected FeatureMap[,] _inGradients;
        protected int _inputDimensions;
        protected FeatureMap[,] _inputs;
        protected ILayerInfo[] _layerInfos;
        protected FeatureMap[,] _outGradients;
        protected int _outputDimensions;
        protected FeatureMap[,] _outputs;
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

        /// <value>The name of the <see cref="Layer"/>, used for logging.</value>
        [JsonIgnore] public abstract string Name { get; }

        /// <value>The number of dimensions in the <see cref="Layer"/>'s output.</value>
        [JsonIgnore] public int OutputDimensions => _outputDimensions;

        /// <value>The output of the latest forward propagation through the <see cref="Layer"/>.</value>
        [JsonIgnore] public FeatureMap[,] Outputs => _outputs;

        /// <summary>
        /// Backpropagates through the <see cref="Layer"/> updating any layer weights, and calculating the outgoing gradient that is
        /// shared with the previous layer.
        /// </summary>
        /// <param name="learningRate">The overall learning rate for the layer updates, corrected for the influence of bias in the first and second moments.</param>
        /// <param name="firstMomentDecay">The exponential decay rate for the first moment.</param>
        /// <param name="secondMomentDecay">The exponential decay rate for the second moment.</param>
        public abstract void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay);

        /// <summary>
        /// Forward propagates through the <see cref="Layer"/> calculating the output <see cref="FeatureMap"/> that is shared with
        /// the next layer.
        /// </summary>
        public abstract void Forward();

        /// <summary>
        /// Reset's the current <see cref="Layer"/> to it's initial weights or initial random weights.
        /// </summary>
        public abstract void Reset();

        /// <summary>
        /// Initializes the <see cref="Layer"/> for the data set being used.
        /// </summary>
        /// <param name="inputs">The previous <see cref="Layer"/>'s output.</param>
        /// <param name="outGradients">The previous <see cref="Layer"/>'s inGradient.</param>
        /// <returns>Returns the output and inGradient to share with the next <see cref="Layer"/>.</returns>
        public abstract (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] inputs, FeatureMap[,] outGradients);

        /// <summary>
        /// Initializes the <see cref="Layer"/> and many of its fields.
        /// </summary>
        /// <param name="inputs">The previous <see cref="Layer"/>'s output.</param>
        /// <param name="outGradients">The previous <see cref="Layer"/>'s inGradient.</param>
        /// <param name="outputDimensionFactor">A factor relating the number of input layers to the number of output layers.
        /// A positive number multiplies the number of input dimensions. A negative number divides the number of dimensions.</param>
        /// <exception cref="ArgumentException">Thrown if the ratio of input layers and output layers is not an integer.</exception>
        protected void BaseStartup(FeatureMap[,] inputs, FeatureMap[,] outGradients, int outputDimensionFactor = 1)
        {
            _inputDimensions = inputs.GetLength(0);
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

            _batchSize = inputs.GetLength(1);
            _layerInfos = new ILayerInfo[_inputDimensions];
            _inputs = inputs;
            _outGradients = outGradients;
            _outputs = new FeatureMap[_outputDimensions, _batchSize];
            _inGradients = new FeatureMap[_outputDimensions, _batchSize];

            for (int i = 0; i < _inputDimensions; i++)
            {
                ILayerInfo layer;
                if (_stride == 1 && _filterSize == 1)
                {
                    layer = _layerInfos[i] = new SingleLayerInfo()
                    {
                        Width = inputs[i, 0].Width,
                        Length = inputs[i, 0].Length,
                    };
                }
                else
                {
                    layer = _layerInfos[i] = new LayerInfo()
                    {
                        FilterSize = _filterSize,
                        Stride = _stride,
                        InverseKSquared = 1f / (_filterSize * _filterSize),
                        InputWidth = inputs[i, 0].Width,
                        InputLength = inputs[i, 0].Length,
                        OutputWidth = 2 + (inputs[i, 0].Width - _filterSize - 1) / _stride,
                        OutputLength = 2 + (inputs[i, 0].Length - _filterSize - 1) / _stride
                    };
                }

                for (int j = 0; j < _batchSize; j++)
                {
                    _outGradients[i, j] = new FeatureMap(layer.InputWidth, layer.InputLength);
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
                    _outputs[i, j] = new FeatureMap(layer.OutputWidth, layer.OutputLength);
                }
            }

            _deviceInputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];
            _deviceInGradients = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions, _batchSize];
            _deviceOutputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions, _batchSize];
            _deviceOutGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions, _batchSize];
        }
    }
}