using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="LatentConvolution"/> class is a <see cref="Layer"/> that performs convolutions based on a set of labels.
    /// Generally used in a <see cref="Networks.Generator"/> to produce images that match the given labels.
    /// </summary>
    public class LatentConvolution : Layer, IPrimaryLayer
    {
        protected const int CLAMP = 1;

        protected const float LEARNINGMULTIPLIER = 1f;

        protected MemoryBuffer1D<float, Stride1D.Dense>[,] _deviceFilterGradients;
        protected MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceFilters;
        protected MemoryBuffer1D<LayerInfo, Stride1D.Dense>[] _deviceInfos;

        protected int _dimensionsMultiplier;

        [JsonProperty] private ColorVector[,] _boolsFilterVectors;
        [JsonProperty] private ColorVector[,] _boolsFirstMoment;
        [JsonProperty] private ColorVector[,] _boolsSecondMoment;

        private ColorTensor[,] _filterGradients;
        private ColorTensor[,] _filters;

        [JsonProperty] private ColorVector[,] _floatsFilterVectors;
        [JsonProperty] private ColorVector[,] _floatsFirstMoment;
        [JsonProperty] private ColorVector[,] _floatsSecondMoment;

        /// <summary>
        /// Initializes a new instance of the <see cref="LatentConvolution"/> class.
        /// </summary>
        /// <param name="filterSize">The width and height of a filter.</param>
        /// <param name="stride">The amount of movement over the image for each filter pass.</param>
        /// <param name="outputDimensionsMultiplier">A factor relating the number of input layers to the number of output layers.
        /// A positive number multiplies the number of input dimensions. A negative number divides the number of dimensions.
        /// Note: Convolution layers are currently only set to increase the number of dimensions.</param>
        public LatentConvolution(int filterSize, int stride, int outputDimensionsMultiplier) : base(filterSize, stride)
        {
            _dimensionsMultiplier = outputDimensionsMultiplier;
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor]
        private LatentConvolution() : base()
        {
        }

        /// <value>A set of bool labels for setting the images filters.</value>
        [JsonIgnore] public bool[][] Bools { get; set; }

        /// <value>A set of float labels for setting the images filters.</value>
        [JsonIgnore] public float[][] Floats { get; set; }

        /// <inheritdoc/>
        public override string Name => "Convolutional Key Layer";

        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            Accelerator accelerator = Utility.Accelerator;

            var backwardsOutKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(Convolution.BackwardsGradientKernal);
            var backwardsGradientKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(Convolution.BackwardsFilterKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = accelerator.Allocate1D(new LayerInfo[] { Infos(i) });
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInputs[i, j] = _inputs[i, j].Allocate(accelerator);
                    _deviceOutGradients[i, j] = _outGradients[i, j].AllocateFloat(accelerator, true);
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                Index3D index = new(Infos(i).OutputWidth, Infos(i).OutputLength, 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceFilters[i, j] = _filters[i,j].Allocate(accelerator);
                    _deviceFilterGradients[i, j] = _filterGradients[i, j].AllocateFloat(accelerator, true);
                    _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);

                    backwardsOutKernal(index, _deviceInGradients[i, j].View, _deviceFilters[i, j].View, _deviceOutGradients[i % _inputDimensions, j].View, _deviceInfos[i % _inputDimensions].View);
                    backwardsGradientKernal(index, _deviceInGradients[i, j].View, _deviceInputs[i % _inputDimensions, j].View, _deviceFilterGradients[i, j].View, _deviceInfos[i % _inputDimensions].View);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _outGradients[i, j].CopyFromBuffer(_deviceOutGradients[i, j]);
                    _deviceOutGradients[i, j].Dispose();
                    _deviceInputs[i, j].Dispose();
                }
                _deviceInfos[i].Dispose();
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInGradients[i, j].Dispose();
                    _filterGradients[i, j].CopyFromBuffer(_deviceFilterGradients[i, j]);
                    _deviceFilterGradients[i, j].Dispose();
                    _deviceFilters[i, j].Dispose();
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    for (int x = 0; x < _filterSize; x++)
                    {
                        for (int y = 0; y < _filterSize; y++)
                        {
                            Color gradient = _filterGradients[i, j][x,y].Clamp(CLAMP);

                            int index = y * _filterSize + x;
                            for (int l = 0; l < _boolsFilterVectors[i, index].Length; l++)
                            {
                                Color gradientColor = Bools[j][l] ? (1 - firstMomentDecay) * gradient : new Color();
                                Color first = _boolsFirstMoment[i, index][l] = firstMomentDecay * _boolsFirstMoment[i, index][l] + gradientColor;
                                Color second = _boolsSecondMoment[i, index][l] = secondMomentDecay * _boolsSecondMoment[i, index][l] + (1 - secondMomentDecay) * Color.Pow(gradientColor, 2);
                                _boolsFilterVectors[i, index][l] -= learningRate * first / (Color.Pow(second, 0.5f) + Utility.AsymptoteErrorColor);
                            }

                            for (int l = 0; l < _floatsFilterVectors[i, index].Length; l++)
                            {
                                Color first = _floatsFirstMoment[i, index][l] = firstMomentDecay * _floatsFirstMoment[i, index][l] + (1 - firstMomentDecay) * gradient * Floats[j][l];
                                Color second = _floatsSecondMoment[i, index][l] = secondMomentDecay * _floatsSecondMoment[i, index][l] + (1 - secondMomentDecay) * Color.Pow(gradient * Floats[j][l], 2);
                                _floatsFilterVectors[i, index][l] -= learningRate * first / (Color.Pow(second, 0.5f) + Utility.AsymptoteErrorColor);
                            }
                        }
                    }
                }
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            //Construct the convolution filters based on the labels then perform the standard Convolution steps.
            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    for (int y = 0; y < _filterSize; y++)
                    {
                        for (int x = 0; x < _filterSize; x++)
                        {
                            _filters[i, j][x,y] = new Color(0);
                            int index = y * _filterSize + x;
                            for (int l = 0; l < _boolsFilterVectors[i, index].Length; l++)
                            {
                                if (Bools[j][l])
                                    _filters[i, j][x,y] += _boolsFilterVectors[i, index][l];
                            }
                            for (int l = 0; l < _floatsFilterVectors[i, index].Length; l++)
                            {
                                _filters[i, j][x, y] += _floatsFilterVectors[i, index][l] * Floats[j][l];
                            }
                        }
                    }
                }
            }

            Accelerator accelerator = Utility.Accelerator;

            var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<LayerInfo>>(Convolution.ForwardKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = accelerator.Allocate1D(new LayerInfo[] { Infos(i) });
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInputs[i, j] = _inputs[i, j].Allocate(accelerator);
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceFilters[i, j] = _filters[i,j].Allocate(accelerator);
                    _deviceOutputs[i, j] = _outputs[i, j].AllocateEmpty(accelerator);

                    forwardKernal(index, _deviceInputs[i % _inputDimensions, j].View, _deviceOutputs[i, j].View, _deviceFilters[i, j].View, _deviceInfos[i % _inputDimensions].View);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _outputs[i, j].CopyFromBuffer(_deviceOutputs[i, j]);
                    _deviceOutputs[i, j].Dispose();
                    _deviceFilters[i, j].Dispose();
                }
            }

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInputs[i, j].Dispose();
                }
                _deviceInfos[i].Dispose();
            }
        }

        /// <summary>
        /// Called when the layer is deserialized.
        /// Temporary function to allow for loading models that were created before Adam optimization was used implemented.
        /// </summary>
        /// <param name="context">The streaming context for deserialization.</param>
        [OnDeserialized]
        public void OnDeserialized(StreamingContext context)
        {
            if (_boolsFirstMoment == null || _boolsSecondMoment == null || _floatsFirstMoment == null || _floatsSecondMoment == null)
            {
                _boolsFirstMoment = new ColorVector[_boolsFilterVectors.GetLength(0), _filterSize * _filterSize];
                _boolsSecondMoment = new ColorVector[_boolsFilterVectors.GetLength(0), _filterSize * _filterSize];

                _floatsFirstMoment = new ColorVector[_floatsFilterVectors.GetLength(0), _filterSize * _filterSize];
                _floatsSecondMoment = new ColorVector[_floatsFilterVectors.GetLength(0), _filterSize * _filterSize];

                for (int i = 0; i < _outputDimensions; i++)
                {
                    for (int j = 0; j < _filterSize * _filterSize; j++)
                    {
                        _boolsFirstMoment[i, j] = new ColorVector(_boolsFilterVectors[i, j].Length);
                        _boolsSecondMoment[i, j] = new ColorVector(_boolsFilterVectors[i, j].Length);

                        _floatsFirstMoment[i, j] = new ColorVector(_floatsFilterVectors[i, j].Length);
                        _floatsSecondMoment[i, j] = new ColorVector(_floatsFilterVectors[i, j].Length);
                    }
                }
            }
        }

        /// <inheritdoc/>
        public override void Reset()
        {
            float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize * (Bools[0].Length + Floats[0].Length) + _inputDimensions * _filterSize * _filterSize * (Bools[0].Length + Floats[0].Length));
            float stdDev = MathF.Sqrt(variance);

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _filterSize * _filterSize; j++)
                {
                    for (int k = 0; k < _boolsFilterVectors[i, j].Length; k++)
                    {
                        _boolsFilterVectors[i, j][k] = Color.RandomGauss(0, stdDev);
                        _boolsFirstMoment[i, j][k] = new Color();
                        _boolsSecondMoment[i, j][k] = new Color();
                    }

                    for (int k = 0; k < _floatsFilterVectors[i, j].Length; k++)
                    {
                        _floatsFilterVectors[i, j][k] = Color.RandomGauss(0, stdDev);

                        _floatsFirstMoment[i, j][k] = new Color();
                        _floatsSecondMoment[i, j][k] = new Color();
                    }
                }
            }
        }

        /// <inheritdoc/>
        public override (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] input, FeatureMap[,] outGradients)
        {
            if (_boolsFilterVectors == null || _floatsFilterVectors == null)
            {
                BaseStartup(input, outGradients, _dimensionsMultiplier);

                _boolsFilterVectors = new ColorVector[_outputDimensions, _filterSize * _filterSize];
                _boolsFirstMoment = new ColorVector[_outputDimensions, _filterSize * _filterSize];
                _boolsSecondMoment = new ColorVector[_outputDimensions, _filterSize * _filterSize];

                _floatsFilterVectors = new ColorVector[_outputDimensions, _filterSize * _filterSize];
                _floatsFirstMoment = new ColorVector[_outputDimensions, _filterSize * _filterSize];
                _floatsSecondMoment = new ColorVector[_outputDimensions, _filterSize * _filterSize];

                float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize * (Bools[0].Length + Floats[0].Length) + _inputDimensions * _filterSize * _filterSize * (Bools[0].Length + Floats[0].Length));
                float stdDev = MathF.Sqrt(variance);

                for (int i = 0; i < _outputDimensions; i++)
                {
                    for (int j = 0; j < _filterSize * _filterSize; j++)
                    {
                        _boolsFilterVectors[i, j] = new ColorVector(Bools[0].Length);
                        _boolsFirstMoment[i, j] = new ColorVector(Bools[0].Length);
                        _boolsSecondMoment[i, j] = new ColorVector(Bools[0].Length);
                        for (int k = 0; k < Bools[0].Length; k++)
                        {
                            _boolsFilterVectors[i, j][k] = Color.RandomGauss(0, stdDev);
                        }

                        _floatsFilterVectors[i, j] = new ColorVector(Floats[0].Length);
                        _floatsFirstMoment[i, j] = new ColorVector(Floats[0].Length);
                        _floatsSecondMoment[i, j] = new ColorVector(Floats[0].Length);
                        for (int k = 0; k < Floats[0].Length; k++)
                        {
                            _floatsFilterVectors[i, j][k] = Color.RandomGauss(0, stdDev);
                        }
                    }
                }
            }
            else
            {
                BaseStartup(input, outGradients, _boolsFilterVectors.GetLength(0) / input.GetLength(0));
                if (_boolsFilterVectors[0, 0].Length != Bools[0].Length || _floatsFilterVectors[0, 0].Length != Floats[0].Length)
                {
                    float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize * (Bools[0].Length + Floats[0].Length) + _inputDimensions * _filterSize * _filterSize * (Bools[0].Length + Floats[0].Length));
                    float stdDev = MathF.Sqrt(variance);
                    for (int i = 0; i < _outputDimensions; i++)
                    {
                        for (int j = 0; j < _filterSize * _filterSize; j++)
                        {
                            ColorVector newBoolVector = new(Bools[0].Length);
                            for (int k = 0; k < Bools[0].Length; k++)
                            {
                                if (_boolsFilterVectors[i, j].Length > k)
                                    newBoolVector[k] = _boolsFilterVectors[i, j][k];
                                else
                                    newBoolVector[k] = Color.RandomGauss(0, stdDev);
                            }
                            _boolsFilterVectors[i, j] = newBoolVector;

                            ColorVector newFloatVector = new(Floats[0].Length);
                            for (int k = 0; k < Floats[0].Length; k++)
                            {
                                if (_floatsFilterVectors[i, j].Length > k)
                                    newFloatVector[k] = _floatsFilterVectors[i, j][k];
                                else
                                    newFloatVector[k] = Color.RandomGauss(0, stdDev);
                            }
                            _floatsFilterVectors[i, j] = newFloatVector;
                        }
                    }
                }
            }

            _filters = new ColorTensor[_outputDimensions, _batchSize];
            _filterGradients = new ColorTensor[_outputDimensions, _batchSize];

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _filters[i, j] = new ColorTensor(_filterSize, _filterSize);
                    _filterGradients[i, j] = new ColorTensor(_filterSize, _filterSize * 3);
                }
            }

            _deviceInfos = new MemoryBuffer1D<LayerInfo, Stride1D.Dense>[_inputDimensions];
            _deviceFilters = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions, _batchSize];
            _deviceFilterGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_outputDimensions, _batchSize];

            return (_outputs, _inGradients);
        }

        /// <summary>
        /// Gets the <see cref="LayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="LayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="LayerInfo"/> corresponding to an input dimension.</returns>
        protected LayerInfo Infos(int index)
        {
            return (LayerInfo)_layerInfos[index % _inputDimensions];
        }
    }
}