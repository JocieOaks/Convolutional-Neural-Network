using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU.Runtime;
using ILGPU;
using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.Layers
{
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

        public static Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<InverseLayerInfo>> BackwardsFilterAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<InverseLayerInfo>>(BackwardsFilterKernel);

        public static Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<InverseLayerInfo>> BackwardsOutGradientAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<InverseLayerInfo>>(BackwardsGradientKernel);

        public static Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<Color>, ArrayView<InverseLayerInfo>> ForwardAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<Color>, ArrayView<InverseLayerInfo>>(ForwardKernel);

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
                Index3D index = new(Infos(i).InputWidth, Infos(i).InputLength, 3);
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
                    GPUManager.CopyAction(index, _buffers.InputsColor[i, j], _inputs[i, j].GetArrayViewEmpty<Color>());
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _buffers.OutputsColor[i, j].SubView(0, Infos(i).OutputArea).MemSetToZero();
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                Index3D index = new(Infos(i).InputWidth, Infos(i).InputLength, 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    ForwardAction(index, _buffers.InputsFloat[i % _inputDimensions, j], _buffers.OutputsFloat[i, j], _filters[i].WeightsGPU<Color>(), _deviceInfos[i % _inputDimensions].View);
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
        public override FeatureMap[,] Startup(FeatureMap[,] inputs, IOBuffers buffers)
        {
            if (_filters == null)
            {
                BaseStartup(inputs, buffers, _dimensionsMultiplier);
                _filters = new Weights[_outputDimensions];

                float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize + _inputDimensions * _filterSize * _filterSize);
                float stdDev = MathF.Sqrt(variance);

                int filterArea = _filterSize * _filterSize;

                for (int i = 0; i < _filters.Length; i++)
                {
                    _filters[i] = new Weights(filterArea, 0, stdDev);
                }
            }
            else
            {
                BaseStartup(inputs, buffers, _filters.Length / inputs.GetLength(0));
            }

            _deviceInfos = new MemoryBuffer1D<InverseLayerInfo, Stride1D.Dense>[_inputDimensions];
            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = GPUManager.Accelerator.Allocate1D(new InverseLayerInfo[] { Infos(i) });
            }

            _inputs = inputs;

            return _outputs;
        }

        /// <summary>
        /// Initializes the <see cref="Layer"/> and many of its fields.
        /// </summary>
        /// <param name="inputs">The previous <see cref="Layer"/>'s output.</param>
        /// <param name="outGradients">The previous <see cref="Layer"/>'s inGradient.</param>
        /// <param name="outputDimensionFactor">A factor relating the number of input layers to the number of output layers.
        /// A positive number multiplies the number of input dimensions. A negative number divides the number of dimensions.</param>
        /// <exception cref="ArgumentException">Thrown if the ratio of input layers and output layers is not an integer.</exception>
        protected new void BaseStartup(FeatureMap[,] inputs, IOBuffers buffers, int outputDimensionFactor = 1)
        {
            _inputDimensions = inputs.GetLength(0);

            _outputDimensions = outputDimensionFactor * _inputDimensions;

            _batchSize = (uint)inputs.GetLength(1);
            _layerInfos = new ILayerInfo[_inputDimensions];
            _outputs = new FeatureMap[_outputDimensions, _batchSize];

            for (int i = 0; i < _inputDimensions; i++)
            {
                _layerInfos[i] = new InverseLayerInfo()
                {
                    FilterSize = _filterSize,
                    Stride = _stride,
                    InverseKSquared = 1f / (_filterSize * _filterSize),
                    InputWidth = inputs[i, 0].Width,
                    InputLength = inputs[i, 0].Length,
                    OutputWidth = inputs[i, 0].Width * _filterSize - (inputs[i, 0].Width - 1) * (_filterSize - _stride),
                    OutputLength = inputs[i, 0].Length * _filterSize - (inputs[i, 0].Length - 1) * (_filterSize - _stride)
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

                for (int j = 0; j < _batchSize; j++)
                {
                    _outputs[i, j] = new FeatureMap(layer.OutputWidth, layer.OutputLength);
                }
            }

            _buffers = buffers;
            for (int i = 0; i < _outputDimensions; i++)
                buffers.OutputDimensionArea(i, _outputs[i, 0].Area);
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
        private static void BackwardsFilterKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> input, ArrayView<float> filterGradient, ArrayView<InverseLayerInfo> info)
        {
            float inputValue = input[3 * info[0].InputIndex(index.X, index.Y) + index.Z];

            for (int j = 0; j < info[0].FilterSize; j++)
            {
                for (int i = 0; i < info[0].FilterSize; i++)
                {
                    int inputIndex = info[0].OutputIndex(index.X, i, index.Y, j);

                    int filterIndex = info[0].FilterIndex(i, j);
                    float dK = inputValue * inGradient[3 * inputIndex + index.Z];
                    Atomic.Add(ref filterGradient[filterIndex * 3 + index.Z], dK);
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
        private static void ForwardKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, ArrayView<Color> filter, ArrayView<InverseLayerInfo> info)
        {
            float dL = input[3 * info[0].InputIndex(index.X, index.Y) + index.Z];

            for (int j = 0; j < info[0].FilterSize; j++)
            {
                for (int i = 0; i < info[0].FilterSize; i++)
                {
                    float dP = dL * filter[info[0].FilterIndex(i, j)][index.Z];
                    Atomic.Add(ref output[info[0].OutputIndex(index.X, i, index.Y, j) * 3 + index.Z], dP);
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
        private static void BackwardsGradientKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<float> filter, ArrayView<InverseLayerInfo> info)
        {
            float sum = 0;

            for (int j = 0; j < info[0].FilterSize; j++)
            {
                for (int i = 0; i < info[0].FilterSize; i++)
                {
                    sum += filter[3 * info[0].FilterIndex(i, j) + index.Z] * inGradient[3 * info[0].OutputIndex(index.X, i, index.Y, j) + index.Z];
                }
            }

            Atomic.Add(ref outGradient[3 * info[0].InputIndex(index.X, index.Y) + index.Z], sum);
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
                    _buffers.OutGradientsColor[i, j].SubView(0, Infos(i).InputArea).MemSetToZero();
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                Index3D index = new(Infos(i).InputWidth, Infos(i).InputLength, 3);
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
                    _buffers.OutGradientsColor[i, j].SubView(0, Infos(i).InputArea).MemSetToZero();
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                Index3D index = new(Infos(i).InputWidth, Infos(i).InputLength, 3);
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
                _filters[i].TestFilterGradient(this, input, _outputs[0, 0], i, _buffers);
            }
        }
    }
}