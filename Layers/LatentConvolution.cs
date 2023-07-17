using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
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
        protected MemoryBuffer1D<LayerInfo, Stride1D.Dense>[] _deviceInfos;
        protected FeatureMap[,] _inputs;

        protected int _dimensionsMultiplier;

        private Vector[,] _filterGradients;
        private Vector[,] _filters;

        [JsonProperty] Weights[,] _boolsFilters;
        [JsonProperty] Weights[,] _floatsFilters;

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
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _buffers.OutGradientsFloat[i, j].SubView(0, Infos(i).InputArea).MemSetToZero();
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
                for (int j = 0; j < _batchSize; j++)
                {
                    Convolution.BackwardsOutGradientAction(index, _buffers.InGradientsFloat[i, j], _filters[i, j].GetArrayView<float>(), _buffers.OutGradientsFloat[i % _inputDimensions, j], _deviceInfos[i % _inputDimensions].View);
                    Convolution.BackwardsFilterAction(index, _buffers.InGradientsFloat[i, j], _inputs[i % _inputDimensions, j].GetArrayView<float>(), _filterGradients[i, j].GetArrayViewZeroed<float>(), _deviceInfos[i % _inputDimensions].View);
                }
            }

            Synchronize();
            DecrementCacheabble(_inputs);
            DecrementCacheabble(_filters);
            DecrementCacheabble(_filterGradients);

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _filterGradients[i, j].SyncCPU();

                    for (int k = 0; k < _filterSize * _filterSize; k++)
                    {
                        float gradient = _filterGradients[i, j][k];

                        for (int l = 0; l < _boolsFilters[i, k].Length; l++)
                        {
                            _boolsFilters[i, k].SetGradient(l, Bools[j][l] ? gradient : 0, learningRate, firstMomentDecay, secondMomentDecay);
                        }

                        for (int l = 0; l < _floatsFilters[i, k].Length; l++)
                        {
                            _floatsFilters[i, k].SetGradient(l, gradient * Floats[j][l], learningRate, firstMomentDecay, secondMomentDecay);
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
                    for (int k = 0; k < _filterSize * _filterSize; k++)
                    {
                        _filters[i, j][k] = 0;
                            
                        for (int l = 0; l < _boolsFilters[i, k].Length; l++)
                        {
                            if (Bools[j][l])
                                _filters[i, j][k] += _boolsFilters[i, k][l];
                        }
                        for (int l = 0; l < _floatsFilters[i, k].Length; l++)
                        {
                            _filters[i, j][k] += _floatsFilters[i, k][l] * Floats[j][l];
                        }

                        _filters[i, j].UpdateIfAllocated();
                    }
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
                for (int j = 0; j < _batchSize; j++)
                {
                    Convolution.ForwardAction(index, _buffers.InputsFloat[i % _inputDimensions, j], _buffers.OutputsFloat[i, j], _filters[i, j].GetArrayView<float>(), _deviceInfos[i % _inputDimensions].View);
                }
            }

            Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index1D index = new(Infos(i).InputArea);
                for (int j = 0; j < _batchSize; j++)
                {
                    GPUManager.CopyAction(index, _buffers.InputsFloat[i, j], _inputs[i, j].GetArrayViewEmpty<float>());
                }
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
                    _boolsFilters[i, j].Reset(0, stdDev);
                    _floatsFilters[i,j].Reset(0, stdDev);
                }
            }
        }

        /// <inheritdoc/>
        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, uint batchSize)
        {
            if (_boolsFilters == null || _floatsFilters == null)
            {
                BaseStartup(inputShapes, buffers, batchSize, _dimensionsMultiplier);

                _boolsFilters = new Weights[_outputDimensions, _filterSize * _filterSize];

                _floatsFilters = new Weights[_outputDimensions, _filterSize * _filterSize];

                float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize * (Bools[0].Length + Floats[0].Length) + _inputDimensions * _filterSize * _filterSize * (Bools[0].Length + Floats[0].Length));
                float stdDev = MathF.Sqrt(variance);

                for (int i = 0; i < _outputDimensions; i++)
                {
                    for (int j = 0; j < _filterSize * _filterSize; j++)
                    {
                        _boolsFilters[i, j] = new Weights(Bools[0].Length, 0, stdDev);
                        _floatsFilters[i, j] = new Weights(Floats[0].Length, 0, stdDev);
                    }
                }
            }
            else
            {
                BaseStartup(inputShapes, buffers, batchSize, _boolsFilters.GetLength(0) / inputShapes.Length);
                if (_boolsFilters[0, 0].Length != Bools[0].Length || _floatsFilters[0, 0].Length != Floats[0].Length)
                {
                    float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize * (Bools[0].Length + Floats[0].Length) + _inputDimensions * _filterSize * _filterSize * (Bools[0].Length + Floats[0].Length));
                    float stdDev = MathF.Sqrt(variance);
                    for (int i = 0; i < _outputDimensions; i++)
                    {
                        for (int j = 0; j < _filterSize * _filterSize; j++)
                        {
                            Weights newBoolVector = new(Bools[0].Length, 0, stdDev);
                            for (int k = 0; k < _boolsFilters[i, j].Length; k++)
                            {
                                newBoolVector.SetWeights(k, _boolsFilters[i, j][k]);
                            }
                            _boolsFilters[i, j] = newBoolVector;

                            Weights newFloatVector = new(Floats[0].Length, 0, stdDev);

                            for (int k = 0; k < _floatsFilters[i, j].Length; k++)
                            {
                                newFloatVector.SetWeights(k, _floatsFilters[i, j][k]);
                            }
                            _floatsFilters[i, j] = newFloatVector;
                        }
                    }
                }
            }

            _filters = new Vector[_outputDimensions, _batchSize];
            _filterGradients = new Vector[_outputDimensions, _batchSize];

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _filters[i, j] = new Vector(_filterSize * _filterSize);
                    _filterGradients[i, j] = new Vector(_filterSize * _filterSize);
                }
            }

            _inputs = new FeatureMap[_inputDimensions, batchSize];
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < batchSize; j++)
                {
                    _inputs[i, j] = new FeatureMap(inputShapes[i]);
                }
            }

            _deviceInfos = new MemoryBuffer1D<LayerInfo, Stride1D.Dense>[_inputDimensions];
            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = GPUManager.Accelerator.Allocate1D(new LayerInfo[] { Infos(i) });
            }

            return _outputShapes;
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

        public void FilterTest(int outputMultiplier)
        {
            FeatureMap[,] input = FilterTestSetup(outputMultiplier);

            for (int j = 0; j < outputMultiplier; j++)
            {
                for (int i = 0; i < _filterSize * _filterSize; i++)
                {
                    FeatureMap output = new(_outputShapes[i]);
                    _boolsFilters[j, i].TestFilterGradient(this, input, output, j, _buffers);
                }

                for (int i = 0; i < _filterSize * _filterSize; i++)
                {
                    FeatureMap output = new(_outputShapes[i]);
                    _floatsFilters[j, i].TestFilterGradient(this, input, output, j, _buffers);
                }
            }
        }
    }
}