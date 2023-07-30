using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ConvolutionalNeuralNetwork.Layers.Initializers;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
namespace ConvolutionalNeuralNetwork.Layers.Weighted
{
    /// <summary>
    /// The <see cref="Convolution"/> class is a <see cref="Layer"/> that performs the titular convolutions of a convolutional
    /// neural network, by passing <see cref="FeatureMap"/>s through a variety of filters.
    /// </summary>
    [Serializable]
    public class Convolution : WeightedLayer, IPrimaryLayer
    {
        private readonly int _outputDimensions;
        [JsonProperty] private Weights _filters;
        private Vector _inputCopy;

        /// <summary>
        /// Initializes a new instance of the <see cref="Convolution"/> layer.
        /// </summary>
        /// <param name="filterSize">The width and height of a filter.</param>
        /// <param name="stride">The amount of movement over the image for each filter pass.</param>
        /// <param name="outputDimensions">A factor relating the number of input layers to the number of output layers.
        /// Must be positive. To reduce the number of output dimensions, use a <see cref="Summation"/> layer afterwards.</param>
        public Convolution(int filterSize, int stride, int outputDimensions, IWeightInitializer initializer, bool useBias = true) : base(filterSize, stride, initializer, useBias)
        {
            if (outputDimensions < 1)
            {
                throw new ArgumentException("Dimension multiplier must be greater than or equal to 1.");
            }
            _outputDimensions = outputDimensions;
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor]
        private Convolution() : base()
        {
        }

        public static Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> BackwardsFilterAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(BackwardsFilterKernel);

        public static Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> BackwardsOutGradientAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(BackwardsGradientKernel);

        public static Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> ForwardAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(ForwardKernel);

        /// <inheritdoc/>
        public override string Name => "Convolutional Layer";

        private LayerInfo Info => (LayerInfo)_layerInfo;

        public void FilterTest(int inputDimensions, int batchSize, int inputSize)
        {
            (Shape input, Shape output) = FilterTestSetup(inputDimensions, batchSize, inputSize);

            _filters.TestFilterGradient(this, input, output, _buffers, batchSize);
            BiasTest(input, output, batchSize);
        }

        /// <inheritdoc/>
        public override void Reset()
        {
            float variance = 2f / (_filterSize * _filterSize);
            float stdDev = MathF.Sqrt(variance);

            for (int i = 0; i < _outputShape.Dimensions; i++)
            {
                _filters.Reset(0, stdDev);
            }
        }

        /// <inheritdoc/>
        public override Shape Startup(Shape inputShapes, IOBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            if (_filters == null)
            {
                BaseStartup(inputShapes, buffers, _outputDimensions);
                _filters = new Weights(_filterSize * _filterSize * _outputShape.Dimensions * _inputShape.Dimensions, _weightInitializer, this);
            }
            else
            {
                BaseStartup(inputShapes, buffers, _filters.Length / _filterSize / _filterSize / inputShapes.Dimensions);
            }

            _inputCopy = new Vector(_inputShape.Dimensions * maxBatchSize * inputShapes.Area);
            return _outputShape;
        }

        /// <summary>
        /// Backpropagates through the layer without updating any of the filter weights. Called when learning rate is zero.
        /// </summary>
        protected override void BackwardsNoUpdate(int batchSize)
        {
            _buffers.OutGradient.SubView(0, batchSize * _inputShape.Volume).MemSetToZero();

            Index3D index = new(_outputShape.Volume, _inputShape.Dimensions, batchSize);
            BackwardsOutGradientAction(index, _buffers.InGradient, _filters.WeightsGPU<float>(), _buffers.OutGradient, Info);

            Synchronize();

            _filters.DecrementLiveWeights();
        }

        /// <summary>
        /// Perform standard backpropagation through the layer, updating it's weights. Called when learning rate is greater than 0.
        /// </summary>
        /// <param name="batchSize"></param>
        /// 
        /// 
        /// 
        protected override void BackwardsUpdate(int batchSize)
        {
            _buffers.OutGradient.SubView(0, batchSize * _inputShape.Volume).MemSetToZero();

            Index3D index = new(_outputShape.Volume, _inputShape.Dimensions, batchSize);
            BackwardsOutGradientAction(index, _buffers.InGradient, _filters.WeightsGPU<float>(), _buffers.OutGradient, Info);
            BackwardsFilterAction(index, _buffers.InGradient, _inputCopy.GetArrayView<float>(), _filters.GradientGPU<float>(), Info);

            Synchronize();

            _inputCopy.DecrementLiveCount();

            _filters.DecrementLiveGradient();
            _filters.DecrementLiveWeights();
            _filters.UpdateWeights(_adamHyperParameters);
        }

        /// <inheritdoc/>
        protected override void ForwardChild(int batchSize)
        {
            Index1D copyIndex = new(_inputShape.Volume * batchSize);
            GPUManager.CopyAction(copyIndex, _buffers.Input, _inputCopy.GetArrayViewEmpty<float>());

            _buffers.Output.SubView(0, batchSize * _outputShape.Volume).MemSetToZero();
            Index3D index = new(_outputShape.Volume, _inputShape.Dimensions, batchSize);
            ForwardAction(index, _buffers.Input, _buffers.Output, _filters.WeightsGPU<float>(), Info);

            Synchronize();

            _inputCopy.DecrementLiveCount();

            _filters.DecrementLiveWeights();
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
        private static void BackwardsFilterKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> input, ArrayView<float> filterGradient, LayerInfo info)
        {
            info.Deconstruct(index.X, index.Y, index.Z, out int mapIndex, out int inputOffset, out int outputIndex, out int dimension);

            float dL = inGradient[outputIndex];

            for (int j = 0; j < info.FilterSize; j++)
            {
                for (int i = 0; i < info.FilterSize; i++)
                {
                    if (info.TryGetInputIndex(mapIndex, i, j, out int inputIndex))
                    {
                        int filterIndex = info.FilterIndex(i, j, dimension);
                        float dK = dL * input[inputIndex + inputOffset];
                        Atomic.Add(ref filterGradient[filterIndex], dK);
                    }
                }
            }
        }

        /// <summary>
        /// An ILGPU kernel to calculate the gradients for backpropagating the previous layer.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
        /// <param name="inGradient">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the incoming
        /// gradient from the following <see cref="Layer"/>.</param>
        /// <param name="filter">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing one of the
        /// <see cref="Convolution"/>'s filters.</param>
        /// <param name="outGradient">An <see cref="ArrayView1D{T, TStride}"/> of floats to sum the outgoing gradient.
        /// Because <see cref="Color"/> cannot be summed atomically, every three floats represents a single
        /// <see cref="Color"/> in the gradient.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void BackwardsGradientKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> filter, ArrayView<float> outGradient, LayerInfo info)
        {
            info.Deconstruct(index.X, index.Y, index.Z, out int mapIndex, out int outGradientOffset, out int inGradientIndex, out int dimension);

            float dL = inGradient[inGradientIndex];

            for (int j = 0; j < info.FilterSize; j++)
            {
                for (int i = 0; i < info.FilterSize; i++)
                {
                    if (info.TryGetInputIndex(mapIndex, i, j, out int outGradientIndex))
                    {
                        int filterIndex = info.FilterIndex(i, j, dimension);
                        float dP = dL * filter[filterIndex];
                        Atomic.Add(ref outGradient[outGradientIndex + outGradientOffset], dP);
                    }
                }
            }
        }

        /// <summary>
        /// An ILGPU kernel for convoluting a <see cref="FeatureMap"/>.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="convoluted">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s to set for the outgoing
        /// convoluted <see cref="FeatureMap"/>.</param>
        /// <param name="filter">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing one of the
        /// <see cref="Convolution"/>'s filters.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void ForwardKernel(Index3D index, ArrayView<float> input, ArrayView<float> convoluted, ArrayView<float> filter, LayerInfo info)
        {
            info.Deconstruct(index.X, index.Y, index.Z, out int mapIndex, out int inputOffset, out int outputIndex, out int dimension);
            float sum = 0;

            for (int j = 0; j < info.FilterSize; j++)
            {
                for (int i = 0; i < info.FilterSize; i++)
        {
                    if (info.TryGetInputIndex(mapIndex, i, j, out int inputIndex))
                        sum += filter[info.FilterIndex(i, j, dimension)] * input[inputIndex + inputOffset];
                }
            }

            Atomic.Add(ref convoluted[outputIndex], sum);
        }
    }
}