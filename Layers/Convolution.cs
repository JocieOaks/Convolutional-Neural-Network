using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="Convolution"/> class is a <see cref="Layer"/> that performs the titular convolutions of a convolutional
    /// neural network, by passing <see cref="FeatureMap"/>s through a variety of filters.
    /// </summary>
    [Serializable]
    public class Convolution : Layer, IPrimaryLayer
    {
        protected const int CLAMP = 1;

        protected const float LEARNINGMULTIPLIER = 1f;

        protected MemoryBuffer1D<float, Stride1D.Dense>[] _deviceFilterGradients;
        protected MemoryBuffer1D<Color, Stride1D.Dense>[] _deviceFilters;
        protected MemoryBuffer1D<LayerInfo, Stride1D.Dense>[] _deviceInfos;
        protected int _dimensionsMultiplier;
        private float[][] _filterGradients;

        [JsonProperty] private Color[][] _filters;

        /// <summary>
        /// Initializes a new instance of the <see cref="Convolution"/> layer.
        /// </summary>
        /// <param name="filterSize">The width and height of a filter.</param>
        /// <param name="stride">The amount of movement over the image for each filter pass.</param>
        /// <param name="outputDimensionsMultiplier">A factor relating the number of input layers to the number of output layers.
        /// A positive number multiplies the number of input dimensions. A negative number divides the number of dimensions.
        /// Note: Convolution layers are currently only set to increase the number of dimensions.</param>
        public Convolution(int filterSize, int stride, int outputDimensionsMultiplier) : base(filterSize, stride)
        {
            _dimensionsMultiplier = outputDimensionsMultiplier;
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor]
        protected Convolution() : base()
        {
        }

        /// <inheritdoc/>
        public override string Name => "Convolutional Layer";

        /// <inheritdoc/>
        public override void Backwards(float learningRate)
        {
            if (learningRate <= 0)
                BackwardsNoUpdate();
            else
                BackwardsUpdate(learningRate);
        }

        /// <summary>
        /// Backpropagates by updating the filters of the <see cref="Convolution"/> layer, but without calculating the gradient for further
        /// propagation. Used in the special case of the first layer of a <see cref="Network"/> to save time.
        /// </summary>
        /// <param name="learningRate">Controls how much the layer is updated with each backpropagation.</param>
        public void BackwardsUpdateOnly(float learningRate)
        {
            Context context = ConvolutionalNeuralNetwork.Utility.Context;
            Accelerator accelerator = ConvolutionalNeuralNetwork.Utility.Accelerator;

            var backwardsGradientKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsFilterKernal);

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
                _deviceFilterGradients[i] = accelerator.Allocate1D<float>(_filterGradients[i].Length);
                Index3D index = new(Infos(i).OutputWidth, Infos(i).OutputLength, 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);

                    backwardsGradientKernal(index, _deviceInGradients[i, j].View, _deviceInputs[i % _inputDimensions, j].View, _deviceFilterGradients[i].View, _deviceInfos[i % _inputDimensions].View);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInputs[i, j].Dispose();
                }
                _deviceInfos[i].Dispose();
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                _deviceFilterGradients[i].CopyToCPU(_filterGradients[i]);
                _deviceFilterGradients[i].Dispose();

                for (int j = 0; j < _filterSize * _filterSize; j++)
                {
                    _filters[i][j] -= learningRate * LEARNINGMULTIPLIER * new Color(_filterGradients[i][j * 3], _filterGradients[i][j * 3 + 1], _filterGradients[i][j * 3 + 2]).Clamp(CLAMP);
                }

                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInGradients[i, j].Dispose();
                }
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            Context context = ConvolutionalNeuralNetwork.Utility.Context;
            Accelerator accelerator = ConvolutionalNeuralNetwork.Utility.Accelerator;

            var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<LayerInfo>>(ForwardKernal);

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
                _deviceFilters[i] = accelerator.Allocate1D(_filters[i]);
                Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceOutputs[i, j] = _outputs[i, j].AllocateEmpty(accelerator);

                    forwardKernal(index, _deviceInputs[i % _inputDimensions, j].View, _deviceOutputs[i, j].View, _deviceFilters[i].View, _deviceInfos[i % _inputDimensions].View);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _outputs[i, j].CopyFromBuffer(_deviceOutputs[i, j]);
                    _deviceOutputs[i, j].Dispose();
                }
                _deviceFilters[i].Dispose();
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

        /// <inheritdoc/>
        public override void Reset()
        {
            float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize + _inputDimensions * _filterSize * _filterSize);
            float stdDev = MathF.Sqrt(variance);

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _filterSize * _filterSize; j++)
                {
                    _filters[i][j] = Color.RandomGauss(0, stdDev);
                }
            }
        }

        /// <inheritdoc/>
        public override (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] input, FeatureMap[,] outGradients)
        {
            if (_filters == null)
            {
                BaseStartup(input, outGradients, _dimensionsMultiplier);
                _filters = new Color[_outputDimensions][];

                float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize + _inputDimensions * _filterSize * _filterSize);
                float stdDev = MathF.Sqrt(variance);

                for (int i = 0; i < _outputDimensions; i++)
                {
                    _filters[i] = new Color[_filterSize * _filterSize];
                    for (int j = 0; j < _filterSize * _filterSize; j++)
                    {
                        _filters[i][j] = Color.RandomGauss(0, stdDev);
                    }
                }
            }
            else
            {
                BaseStartup(input, outGradients, _filters.Length / input.GetLength(0));
            }
            _filterGradients = new float[_outputDimensions][];

            for (int i = 0; i < _outputDimensions; i++)
            {
                _filterGradients[i] = new float[_filterSize * _filterSize * 3];
            }

            _deviceInfos = new MemoryBuffer1D<LayerInfo, Stride1D.Dense>[_inputDimensions];
            _deviceFilters = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions];
            _deviceFilterGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_outputDimensions];

            return (_outputs, _inGradients);
        }

        /// <summary>
        /// An ILGPU kernal to update the <see cref="Convolution"/>'s filters.
        /// </summary>
        /// <param name="index">The index of the current kernal calculation to be made.</param>
        /// <param name="inGradient">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the incoming
        /// gradient from the following <see cref="Layer"/>.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="filterGradient">An <see cref="ArrayView1D{T, TStride}"/> of floats to sum the gradient of the filters.
        /// Because <see cref="Color"/> cannot be summed atomically, every three floats represents a single
        /// <see cref="Color"/> in the gradient.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        protected static void BackwardsFilterKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<Color> input, ArrayView<float> filterGradient, ArrayView<LayerInfo> info)
        {
            float dL = inGradient[info[0].OutputIndex(index.X, index.Y)][index.Z] * info[0].InverseKSquared;

            for (int j = 0; j < info[0].FilterSize; j++)
            {
                for (int i = 0; i < info[0].FilterSize; i++)
                {
                    if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                    {
                        int filterIndex = info[0].FilterIndex(i, j);
                        float dK = dL * input[inputIndex][index.Z];
                        Atomic.Add(ref filterGradient[filterIndex * 3 + index.Z], dK);
                    }
                }
            }
        }

        /// <summary>
        /// An ILGPU kernal to calculate the gradients for backpropagating the previous layer.
        /// </summary>
        /// <param name="index">The index of the current kernal calculation to be made.</param>
        /// <param name="inGradient">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the incoming
        /// gradient from the following <see cref="Layer"/>.</param>
        /// <param name="filter">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing one of the
        /// <see cref="Convolution"/>'s filters.</param>
        /// <param name="outGradient">An <see cref="ArrayView1D{T, TStride}"/> of floats to sum the outgoing gradient.
        /// Because <see cref="Color"/> cannot be summed atomically, every three floats represents a single
        /// <see cref="Color"/> in the gradient.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        protected static void BackwardsGradientKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<Color> filter, ArrayView<float> outGradient, ArrayView<LayerInfo> info)
        {
            float dL = inGradient[info[0].OutputIndex(index.X, index.Y)][index.Z] * info[0].InverseKSquared;

            for (int j = 0; j < info[0].FilterSize; j++)
            {
                for (int i = 0; i < info[0].FilterSize; i++)
                {
                    if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                    {
                        int filterIndex = info[0].FilterIndex(i, j);
                        float dP = dL * filter[filterIndex][index.Z];
                        Atomic.Add(ref outGradient[inputIndex * 3 + index.Z], dP);
                    }
                }
            }
        }

        /// <summary>
        /// An ILGPU kernal to convolute a <see cref="FeatureMap"/>.
        /// </summary>
        /// <param name="index">The index of the current kernal calculation to be made.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="convoluted">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s to set for the outgoing
        /// convoluted <see cref="FeatureMap"/>.</param>
        /// <param name="filter">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing one of the
        /// <see cref="Convolution"/>'s filters.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        protected static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> convoluted, ArrayView<Color> filter, ArrayView<LayerInfo> info)
        {
            Color sum = new();

            for (int j = 0; j < info[0].FilterSize; j++)
            {
                for (int i = 0; i < info[0].FilterSize; i++)
                {
                    if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                        sum += filter[info[0].FilterIndex(i, j)] * input[inputIndex];
                }
            }

            convoluted[info[0].OutputIndex(index.X, index.Y)] = sum * info[0].InverseKSquared;
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

        /// <summary>
        /// Backpropagates through the layer without updating any of the filter weights. Called when learning rate is zero.
        /// </summary>
        private void BackwardsNoUpdate()
        {
            Context context = ConvolutionalNeuralNetwork.Utility.Context;
            Accelerator accelerator = ConvolutionalNeuralNetwork.Utility.Accelerator;

            var backwardsOutKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsGradientKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = accelerator.Allocate1D(new LayerInfo[] { Infos(i) });
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceOutGradients[i, j] = _outGradients[i, j].AllocateFloat(accelerator);
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                _deviceFilters[i] = accelerator.Allocate1D(_filters[i]);
                Index3D index = new(Infos(i).OutputWidth, Infos(i).OutputLength, 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);

                    backwardsOutKernal(index, _deviceInGradients[i, j].View, _deviceFilters[i].View, _deviceOutGradients[i % _inputDimensions, j].View, _deviceInfos[i % _inputDimensions].View);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _outGradients[i, j].CopyFromBuffer(_deviceOutGradients[i, j]);
                    _deviceOutGradients[i, j].Dispose();
                }
                _deviceInfos[i].Dispose();
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                _deviceFilters[i].Dispose();

                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInGradients[i, j].Dispose();
                }
            }
        }

        /// <summary>
        /// Perform standard backpropagation through the layer, updating it's weights. Called when learning rate is greater than 0.
        /// </summary>
        /// <param name="learningRate">Controls how much the layer is updated with each backpropagation.</param>
        private void BackwardsUpdate(float learningRate)
        {
            Context context = ConvolutionalNeuralNetwork.Utility.Context;
            Accelerator accelerator = ConvolutionalNeuralNetwork.Utility.Accelerator;

            var backwardsOutKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsGradientKernal);
            var backwardsGradientKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsFilterKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = accelerator.Allocate1D(new LayerInfo[] { Infos(i) });
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInputs[i, j] = _inputs[i, j].Allocate(accelerator);
                    _deviceOutGradients[i, j] = _outGradients[i, j].AllocateFloat(accelerator);
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                _deviceFilters[i] = accelerator.Allocate1D(_filters[i]);
                _deviceFilterGradients[i] = accelerator.Allocate1D<float>(_filterGradients[i].Length);
                Index3D index = new(Infos(i).OutputWidth, Infos(i).OutputLength, 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);

                    backwardsOutKernal(index, _deviceInGradients[i, j].View, _deviceFilters[i].View, _deviceOutGradients[i % _inputDimensions, j].View, _deviceInfos[i % _inputDimensions].View);
                    backwardsGradientKernal(index, _deviceInGradients[i, j].View, _deviceInputs[i % _inputDimensions, j].View, _deviceFilterGradients[i].View, _deviceInfos[i % _inputDimensions].View);
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
                _deviceFilterGradients[i].CopyToCPU(_filterGradients[i]);
                _deviceFilterGradients[i].Dispose();
                _deviceFilters[i].Dispose();

                for (int j = 0; j < _filterSize * _filterSize; j++)
                {
                    _filters[i][j] -= learningRate * LEARNINGMULTIPLIER * new Color(_filterGradients[i][j * 3], _filterGradients[i][j * 3 + 1], _filterGradients[i][j * 3 + 2]).Clamp(CLAMP);
                }

                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInGradients[i, j].Dispose();
                }
            }
        }
    }
}