using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU.Runtime;
using ILGPU;
using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.Layers
{
    [Serializable]
    public class TransposeConvolution : Layer, IPrimaryLayer
    {
        private MemoryBuffer1D<InverseLayerInfo, Stride1D.Dense>[] _deviceInfos;
        private FeatureMap[,] _inputs;
        private readonly int _dimensionsMultiplier;
        [JsonProperty] private Weights[] _filters;

        /// <summary>
        /// Initializes a new instance of the <see cref="Convolution"/> layer.
        /// </summary>
        /// <param name="filterSize">The width and height of a filter.</param>
        /// <param name="stride">The amount of movement over the image for each filter pass.</param>
        /// <param name="outputDimensionsMultiplier">A factor relating the number of input layers to the number of output layers.
        /// Must be positive. To reduce the number of output dimensions, use a <see cref="Summation"/> layer afterwards.</param>
        public TransposeConvolution(int filterSize, int stride, int outputDimensionsMultiplier) : base(filterSize, stride)
        {
            if (outputDimensionsMultiplier < 1)
            {
                throw new ArgumentException("Dimension multiplier must be greater than or equal to 1.");
            }
            _dimensionsMultiplier = outputDimensionsMultiplier;
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor]
        private TransposeConvolution() : base()
        {
        }

        public static Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<InverseLayerInfo>> BackwardsFilterAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<InverseLayerInfo>>(BackwardsFilterKernel);

        public static Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<InverseLayerInfo>> BackwardsOutGradientAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<InverseLayerInfo>>(BackwardsGradientKernel);

        public static Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<InverseLayerInfo>> ForwardAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<InverseLayerInfo>>(ForwardKernel);

        /// <inheritdoc/>
        public override string Name => "Convolutional Layer";

        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            if (learningRate <= 0)
                BackwardsNoUpdate();
            else
                BackwardsUpdate(learningRate, firstMomentDecay, secondMomentDecay);
        }

        /// <summary>
        /// Backpropagates by updating the filters of the <see cref="Convolution"/> layer, but without calculating the gradient for further
        /// propagation. Used in the special case of the first layer of a <see cref="Network"/> to save time.
        /// </summary>
        /// <param name="learningRate">Controls how much the layer is updated with each backpropagation.</param>
        public void BackwardsUpdateOnly(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            for (int i = 0; i < _outputDimensions; i++)
            {
                Index2D index = new(Infos(i).InputWidth, Infos(i).InputLength);
                for (int j = 0; j < _batchSize; j++)
                {
                    BackwardsFilterAction(index, _buffers.InGradientsFloat[i, j], _inputs[i % _inputDimensions, j].GetArrayView<float>(), _filters[i].GradientGPU<float>(), _deviceInfos[i % _inputDimensions].View);
                }
            }

            Synchronize();
            DecrementCacheabble(_inputs, (uint)(_outputDimensions / _inputDimensions));

            for (int i = 0; i < _outputDimensions; i++)
            {
                _filters[i].DisposeGradient(_batchSize);
                _filters[i].UpdateWeights(learningRate, firstMomentDecay, secondMomentDecay);
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index1D index = new(Infos(i).InputArea);
                for (int j = 0; j < _batchSize; j++)
                {
                    GPUManager.CopyAction(index, _buffers.InputsFloat[i, j], _inputs[i, j].GetArrayViewEmpty<float>());
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _buffers.OutputsFloat[i, j].SubView(0, Infos(i).OutputArea).MemSetToZero();
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                Index2D index = new(Infos(i).InputWidth, Infos(i).InputLength);
                for (int j = 0; j < _batchSize; j++)
                {
                    ForwardAction(index, _buffers.InputsFloat[i % _inputDimensions, j], _buffers.OutputsFloat[i, j], _filters[i].WeightsGPU<float>(), _deviceInfos[i % _inputDimensions].View);
                }
            }

            Synchronize();
            DecrementCacheabble(_inputs);

            for (int i = 0; i < _outputDimensions; i++)
            {
                _filters[i].DisposeWeights(_batchSize);
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
            float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize + _inputDimensions * _filterSize * _filterSize);
            float stdDev = MathF.Sqrt(variance);

            for (int i = 0; i < _outputDimensions; i++)
            {
                _filters[i].Reset(0, stdDev);
            }
        }

        /// <inheritdoc/>
        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, uint batchSize)
        {
            if (_filters == null)
            {
                BaseStartup(inputShapes, buffers, batchSize, _dimensionsMultiplier);
                _filters = new Weights[_outputDimensions];

                float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize + _inputDimensions * _filterSize * _filterSize);
                float stdDev = MathF.Sqrt(variance);

                int filterArea = _filterSize * _filterSize;

                for (int i = 0; i < _filters.Length; i++)
                {
                    _filters[i] = new Weights(filterArea, 0, 0.02f);
                }
            }
            else
            {
                BaseStartup(inputShapes, buffers, batchSize, _filters.Length / inputShapes.GetLength(0));
            }

            _deviceInfos = new MemoryBuffer1D<InverseLayerInfo, Stride1D.Dense>[_inputDimensions];
            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = GPUManager.Accelerator.Allocate1D(new InverseLayerInfo[] { Infos(i) });
            }

            _inputs = new FeatureMap[_inputDimensions, batchSize];
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < batchSize; j++)
                {
                    _inputs[i, j] = new FeatureMap(inputShapes[i]);
                }
            }

            return _outputShapes;
        }

        /// <summary>
        /// Initializes the <see cref="Layer"/> and many of its fields.
        /// </summary>
        /// <param name="inputShapes">The previous <see cref="Layer"/>'s output.</param>
        /// <param name="outGradients">The previous <see cref="Layer"/>'s inGradient.</param>
        /// <param name="outputDimensionFactor">A factor relating the number of input layers to the number of output layers.
        /// A positive number multiplies the number of input dimensions. A negative number divides the number of dimensions.</param>
        /// <exception cref="ArgumentException">Thrown if the ratio of input layers and output layers is not an integer.</exception>
        protected new void BaseStartup(Shape[] inputShapes, IOBuffers buffers, uint batchSize, int outputDimensionFactor = 1)
        {
            _inputDimensions = inputShapes.GetLength(0);

            _outputDimensions = outputDimensionFactor * _inputDimensions;

            _batchSize = batchSize;
            _layerInfos = new ILayerInfo[_inputDimensions];
            _outputShapes = new Shape[_outputDimensions];

            for (int i = 0; i < _inputDimensions; i++)
            {
                _layerInfos[i] = new InverseLayerInfo()
                {
                    FilterSize = _filterSize,
                    Stride = _stride,
                    InverseKSquared = 1f / (_filterSize * _filterSize),
                    InputWidth = inputShapes[i].Width,
                    InputLength = inputShapes[i].Length,
                    OutputWidth = inputShapes[i].Width * _stride,
                    OutputLength = inputShapes[i].Length * _stride,
                    Padding = _filterSize - _stride
                };
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

                _outputShapes[i] = new Shape(layer.OutputWidth, layer.OutputLength);
            }

            _buffers = buffers;
            for (int i = 0; i < _outputDimensions; i++)
                buffers.OutputDimensionArea(i, _outputShapes[i].Area);
        }

        /// <summary>
        /// Gets the <see cref="LayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="LayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="LayerInfo"/> corresponding to an input dimension.</returns>
        private InverseLayerInfo Infos(int index)
        {
            return (InverseLayerInfo)_layerInfos[index % _inputDimensions];
        }

        /// <summary>
        /// An ILGPU kernel to update the <see cref="Convolution"/>'s filters.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
        /// <param name="inGradient">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the incoming
        /// gradient from the following <see cref="Layer"/>.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="filterGradient">An <see cref="ArrayView1D{T, TStride}"/> of floats to sum the gradient of the filters.
        /// Because <see cref="Color"/> cannot be summed atomically, every three floats represents a single
        /// <see cref="Color"/> in the gradient.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void BackwardsFilterKernel(Index2D index, ArrayView<float> inGradient, ArrayView<float> input, ArrayView<float> filterGradient, ArrayView<InverseLayerInfo> info)
        {
            float inputValue = input[info[0].InputIndex(index.X, index.Y)];

            for (int j = 0; j < info[0].FilterSize; j++)
            {
                for (int i = 0; i < info[0].FilterSize; i++)
                {
                    if (info[0].TryGetOutputIndex(index.X, i, index.Y, j, out int outputIndex))
                    {
                        int filterIndex = info[0].FilterIndex(i, j);
                        float dK = inputValue * inGradient[outputIndex];
                        Atomic.Add(ref filterGradient[filterIndex], dK);
                    }
                }
            }
        }

        /// <summary>
        /// An ILGPU kernel to calculate the gradients for backpropagating the previous layer.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the incoming
        /// gradient from the following <see cref="Layer"/>.</param>
        /// <param name="filter">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing one of the
        /// <see cref="Convolution"/>'s filters.</param>
        /// <param name="output">An <see cref="ArrayView1D{T, TStride}"/> of floats to sum the outgoing gradient.
        /// Because <see cref="Color"/> cannot be summed atomically, every three floats represents a single
        /// <see cref="Color"/> in the gradient.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void ForwardKernel(Index2D index, ArrayView<float> input, ArrayView<float> output, ArrayView<float> filter, ArrayView<InverseLayerInfo> info)
        {
            float dL = input[info[0].InputIndex(index.X, index.Y)];

            for (int j = 0; j < info[0].FilterSize; j++)
            {
                for (int i = 0; i < info[0].FilterSize; i++)
                {
                    if (info[0].TryGetOutputIndex(index.X, i, index.Y, j, out int outputIndex))
                    {
                        float dP = dL * filter[info[0].FilterIndex(i, j)];
                        Atomic.Add(ref output[outputIndex], dP);
                    }
                }
            }
        }

        /// <summary>
        /// An ILGPU kernel for convoluting a <see cref="FeatureMap"/>.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
        /// <param name="inGradient">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="outGradient">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s to set for the outgoing
        /// convoluted <see cref="FeatureMap"/>.</param>
        /// <param name="filter">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing one of the
        /// <see cref="Convolution"/>'s filters.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void BackwardsGradientKernel(Index2D index, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<float> filter, ArrayView<InverseLayerInfo> info)
        {
            float sum = 0;

            for (int j = 0; j < info[0].FilterSize; j++)
            {
                for (int i = 0; i < info[0].FilterSize; i++)
                {
                    if(info[0].TryGetOutputIndex(index.X, i, index.Y, j, out int outputIndex))
                        sum += filter[info[0].FilterIndex(i, j)] * inGradient[outputIndex];
                }
            }

            Atomic.Add(ref outGradient[info[0].InputIndex(index.X, index.Y)], sum);
        }

        /// <summary>
        /// Backpropagates through the layer without updating any of the filter weights. Called when learning rate is zero.
        /// </summary>
        private void BackwardsNoUpdate()
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
                Index2D index = new(Infos(i).InputWidth, Infos(i).InputLength);
                for (int j = 0; j < _batchSize; j++)
                {
                    BackwardsOutGradientAction(index, _buffers.InGradientsFloat[i, j], _buffers.OutGradientsFloat[i % _inputDimensions, j], _filters[i].WeightsGPU<float>(), _deviceInfos[i % _inputDimensions].View);
                }
            }
            Synchronize();

            for (int i = 0; i < _outputDimensions; i++)
            {
                _filters[i].DisposeWeights(_batchSize);
            }
        }

        /// <summary>
        /// Perform standard backpropagation through the layer, updating it's weights. Called when learning rate is greater than 0.
        /// </summary>
        /// <param name="learningRate">The overall learning rate for the layer updates, corrected for the influence of bias in the first and second moments.</param>
        /// <param name="firstMomentDecay">The exponential decay rate for the first moment.</param>
        /// <param name="secondMomentDecay">The exponential decay rate for the second moment.</param>
        private void BackwardsUpdate(float learningRate, float firstMomentDecay, float secondMomentDecay)
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
                Index2D index = new(Infos(i).InputWidth, Infos(i).InputLength);
                for (int j = 0; j < _batchSize; j++)
                {
                    BackwardsOutGradientAction(index, _buffers.InGradientsFloat[i, j], _buffers.OutGradientsFloat[i % _inputDimensions, j], _filters[i].WeightsGPU<float>(), _deviceInfos[i % _inputDimensions].View);
                    BackwardsFilterAction(index, _buffers.InGradientsFloat[i, j], _inputs[i % _inputDimensions, j].GetArrayView<float>(), _filters[i].GradientGPU<float>(), _deviceInfos[i % _inputDimensions].View);
                }
            }

            Synchronize();
            DecrementCacheabble(_inputs, (uint)(_outputDimensions / _inputDimensions));

            for (int i = 0; i < _outputDimensions; i++)
            {
                _filters[i].DisposeGradient(_batchSize);
                _filters[i].DisposeWeights(_batchSize);
                _filters[i].UpdateWeights(learningRate, firstMomentDecay, secondMomentDecay);
            }
        }

        public void FilterTest(int outputMultiplier)
        {
            FeatureMap[,] input = FilterTestSetup(outputMultiplier);
            for (int i = 0; i < outputMultiplier; i++)
            {
                FeatureMap output = new(_outputShapes[i]);
                _filters[i].TestFilterGradient(this, input, output, i, _buffers);
            }
        }
    }
}