using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="Convolution"/> class is a <see cref="Layer"/> that performs the titular convolutions of a convolutional
    /// neural network, by passing <see cref="FeatureMap"/>s through a variety of filters.
    /// </summary>
    [Serializable]
    public class Convolution : Layer, IPrimaryLayer
    {
        private MemoryBuffer1D<LayerInfo, Stride1D.Dense>[] _deviceInfos;
        private MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceInputs;
        private readonly int _dimensionsMultiplier;
        private FeatureMap[,] _inputs;
        [JsonProperty] private Weights[] _filters;

        /// <summary>
        /// Initializes a new instance of the <see cref="Convolution"/> layer.
        /// </summary>
        /// <param name="filterSize">The width and height of a filter.</param>
        /// <param name="stride">The amount of movement over the image for each filter pass.</param>
        /// <param name="outputDimensionsMultiplier">A factor relating the number of input layers to the number of output layers.
        /// Must be positive. To reduce the number of output dimensions, use a <see cref="Summation"/> layer afterwards.</param>
        public Convolution(int filterSize, int stride, int outputDimensionsMultiplier) : base(filterSize, stride)
        {
            if(outputDimensionsMultiplier < 1)
            {
                throw new ArgumentException("Dimension multiplier must be greater than or equal to 1.");
            }
            _dimensionsMultiplier = outputDimensionsMultiplier;
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor]
        private Convolution() : base()
        {
        }

        public static Action<Index3D, ArrayView<float>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>> BackwardsFilterAction { get; } = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsFilterKernal);

        public static Action<Index3D, ArrayView<float>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>> BackwardsOutGradientAction { get; } = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsGradientKernal);

        public static Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<LayerInfo>> ForwardAction { get; } = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<LayerInfo>>(ForwardKernal);

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
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInputs[i, j] = _inputs[i, j].Allocate(Utility.Accelerator);
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                Index3D index = new(Infos(i).OutputWidth, Infos(i).OutputLength, 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    BackwardsFilterAction(index, _buffers.InGradientsFloat[i, j], _deviceInputs[i % _inputDimensions, j].View, _filters[i].GradientGPU(), _deviceInfos[i % _inputDimensions].View);
                }
            }

            Utility.Accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInputs[i, j].Dispose();
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                _filters[i].DisposeGradient();
                _filters[i].UpdateFilter(learningRate, firstMomentDecay, secondMomentDecay);
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            for (int i = 0; i < _outputDimensions; i++)
            {
                Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
                for (int j = 0; j < _batchSize; j++)
                {
                    ForwardAction(index, _buffers.InputsColor[i % _inputDimensions, j], _buffers.OutputsColor[i, j], _filters[i].FilterGPU(), _deviceInfos[i % _inputDimensions].View);
                }
            }

            Utility.Accelerator.Synchronize();

            for (int i = 0; i < _outputDimensions; i++)
            {
                _filters[i].DisposeFilter();
            }

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _inputs[i, j].CopyFromBuffer(_buffers.InputsColor[i, j]);
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

            _inputs = inputs;

            _deviceInfos = new MemoryBuffer1D<LayerInfo, Stride1D.Dense>[_inputDimensions];
            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = Utility.Accelerator.Allocate1D(new LayerInfo[] { Infos(i) });
            }
            _deviceInputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];

            return _outputs;
        }

        /// <summary>
        /// Gets the <see cref="LayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="LayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="LayerInfo"/> corresponding to an input dimension.</returns>
        private LayerInfo Infos(int index)
        {
            return (LayerInfo)_layerInfos[index % _inputDimensions];
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
        private static void BackwardsFilterKernal(Index3D index, ArrayView<float> inGradient, ArrayView<Color> input, ArrayView<float> filterGradient, ArrayView<LayerInfo> info)
        {
            float dL = inGradient[3 * info[0].OutputIndex(index.X, index.Y) + index.Z] * info[0].InverseKSquared;

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
        private static void BackwardsGradientKernal(Index3D index, ArrayView<float> inGradient, ArrayView<Color> filter, ArrayView<float> outGradient, ArrayView<LayerInfo> info)
        {
            float dL = inGradient[3 * info[0].OutputIndex(index.X, index.Y) + index.Z] * info[0].InverseKSquared;

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
        /// An ILGPU kernal for convoluting a <see cref="FeatureMap"/>.
        /// </summary>
        /// <param name="index">The index of the current kernal calculation to be made.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="convoluted">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s to set for the outgoing
        /// convoluted <see cref="FeatureMap"/>.</param>
        /// <param name="filter">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing one of the
        /// <see cref="Convolution"/>'s filters.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> convoluted, ArrayView<Color> filter, ArrayView<LayerInfo> info)
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
        /// Backpropagates through the layer without updating any of the filter weights. Called when learning rate is zero.
        /// </summary>
        private void BackwardsNoUpdate()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                for(int j = 0; j < _batchSize; j++)
                {
                    _buffers.OutGradientsColor[i, j].SubView(0, Infos(i).InputArea).MemSetToZero();
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                Index3D index = new(Infos(i).OutputWidth, Infos(i).OutputLength, 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    BackwardsOutGradientAction(index, _buffers.InGradientsFloat[i, j], _filters[i].FilterGPU(), _buffers.OutGradientsFloat[i % _inputDimensions, j], _deviceInfos[i % _inputDimensions].View);
                }
            }

            Utility.Accelerator.Synchronize();

            for (int i = 0; i < _outputDimensions; i++)
            {
                _filters[i].DisposeFilter();
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
                    _deviceInputs[i, j] = _inputs[i, j].Allocate(Utility.Accelerator);
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                Index3D index = new(Infos(i).OutputWidth, Infos(i).OutputLength, 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    BackwardsOutGradientAction(index, _buffers.InGradientsFloat[i, j], _filters[i].FilterGPU(), _buffers.OutGradientsFloat[i % _inputDimensions, j], _deviceInfos[i % _inputDimensions].View);
                    BackwardsFilterAction(index, _buffers.InGradientsFloat[i, j], _deviceInputs[i % _inputDimensions, j].View, _filters[i].GradientGPU(), _deviceInfos[i % _inputDimensions].View);
                }
            }

            Utility.Accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInputs[i, j].Dispose();
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                _filters[i].DisposeGradient();
                _filters[i].DisposeFilter();
                _filters[i].UpdateFilter(learningRate, firstMomentDecay, secondMomentDecay);
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