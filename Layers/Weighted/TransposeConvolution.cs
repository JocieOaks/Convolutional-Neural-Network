using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU.Runtime;
using ILGPU;
using Newtonsoft.Json;
using System.Runtime.Serialization;
using System.Reflection.Emit;
using ConvolutionalNeuralNetwork.Layers.Initializers;

namespace ConvolutionalNeuralNetwork.Layers.Weighted
{
    [Serializable]
    public class TransposeConvolution : WeightedLayer, IPrimaryLayer
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
        public TransposeConvolution(int filterSize, int stride, int outputDimensions, IWeightInitializer initializer, bool useBias = true) : base(filterSize, stride, initializer, useBias)
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
        private TransposeConvolution() : base()
        {
        }

        public static Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, InverseLayerInfo> BackwardsFilterAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, InverseLayerInfo>(BackwardsFilterKernel);

        public static Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, InverseLayerInfo> BackwardsOutGradientAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, InverseLayerInfo>(BackwardsGradientKernel);

        public static Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, InverseLayerInfo> ForwardAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, InverseLayerInfo>(ForwardKernel);

        /// <inheritdoc/>
        public override string Name => "Transpose Convolutional Layer";

        private InverseLayerInfo Info => (InverseLayerInfo)_layerInfo;

        public void FilterTest(int inputDimensions, int batchSize, int inputSize)
        {
            (Shape input, Shape output) = FilterTestSetup(inputDimensions, batchSize, inputSize);

            _filters.TestFilterGradient(this, input, output, _buffers, batchSize);
            BiasTest(input, output, batchSize);
        }

        /// <inheritdoc/>
        protected override void ForwardChild(int batchSize)
        {

            Index1D copyIndex = new(batchSize * _inputShape.Volume);
            GPUManager.CopyAction(copyIndex, _buffers.Input, _inputCopy.GetArrayViewEmpty<float>());

            _buffers.Output.SubView(0, batchSize * _outputShape.Volume).MemSetToZero();

            Index3D index = new(batchSize, _outputShape.Dimensions * _inputShape.Dimensions, _inputShape.Area);
            ForwardAction(index, _buffers.Input, _buffers.Output, _filters.WeightsGPU<float>(), Info);

            Synchronize();

            _inputCopy.DecrementLiveCount();

            _filters.DecrementLiveWeights();
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
            float variance = 0.6666f / (_outputShape.Dimensions * _filterSize * _filterSize + _inputShape.Dimensions * _filterSize * _filterSize);
            float stdDev = MathF.Sqrt(variance);

            _filters.Reset(0, 0.02f);
        }
        /// <inheritdoc/>
        public override Shape Startup(Shape inputShape, IOBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            if (_filters == null)
            {
                BaseStartup(inputShape, buffers, _outputDimensions);

                int filterArea = _filterSize * _filterSize * _outputShape.Dimensions * _inputShape.Dimensions;
                _filters = new Weights(filterArea, _weightInitializer, this);

            }
            else
            {
                BaseStartup(inputShape, buffers, _filters.Length / _filterSize / _filterSize / inputShape.Dimensions);
            }

            _inputCopy = new Vector(_inputShape.Dimensions * maxBatchSize * _inputShape.Area);

            return _outputShape;
        }

        /// <summary>
        /// Initializes the <see cref="Layer"/> and many of its fields.
        /// </summary>
        /// <param name="inputShape">The previous <see cref="Layer"/>'s output.</param>
        /// <param name="outGradients">The previous <see cref="Layer"/>'s inGradient.</param>
        /// <param name="outputDimensions">A factor relating the number of input layers to the number of output layers.
        /// A positive number multiplies the number of input dimensions. A negative number divides the number of dimensions.</param>
        /// <exception cref="ArgumentException">Thrown if the ratio of input layers and output layers is not an integer.</exception>
        protected new void BaseStartup(Shape inputShape, IOBuffers buffers, int outputDimensions = 1)
        {
            _inputShape = inputShape;
            _outputShape = new Shape(inputShape.Width * _stride, inputShape.Length * _stride, outputDimensions);

            _layerInfo = new InverseLayerInfo(inputShape, _outputShape, _filterSize, _stride);

            _buffers = buffers;
            buffers.OutputDimensionArea(_outputShape.Volume);
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
        private static void BackwardsFilterKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> input, ArrayView<float> filterGradient, InverseLayerInfo info)
        {
            (int inputOffset, int inGradientOffset) = info.GetOffset(index.X, index.Y);

            float inputValue = input[index.Z + inputOffset];

            for (int j = 0; j < info.FilterSize; j++)
            {
                for (int i = 0; i < info.FilterSize; i++)
                {
                    if (info.TryGetOutputIndex(index.Z, i, j, out int outputIndex))
                    {
                        int filterIndex = info.FilterIndex(i, j, index.Y);
                        float dK = inputValue * inGradient[outputIndex + inGradientOffset];
                        Atomic.Add(ref filterGradient[filterIndex], dK);
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
        private static void BackwardsGradientKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<float> filter, InverseLayerInfo info)
        {
            (int outGradientOffset, int inGradientOffset) = info.GetOffset(index.X, index.Y);
            float sum = 0;

            for (int j = 0; j < info.FilterSize; j++)
            {
                for (int i = 0; i < info.FilterSize; i++)
                {
                    if (info.TryGetOutputIndex(index.Z, i, j, out int outputIndex))
                        sum += filter[info.FilterIndex(i, j, index.Y)] * inGradient[outputIndex + inGradientOffset];
                }
            }

            Atomic.Add(ref outGradient[index.Z + outGradientOffset], sum);
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
        private static void ForwardKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, ArrayView<float> filter, InverseLayerInfo info)
        {
            (int inputOffset, int outputOffset) = info.GetOffset(index.X, index.Y);
            float dL = input[index.Z + inputOffset];

            for (int j = 0; j < info.FilterSize; j++)
            {
                for (int i = 0; i < info.FilterSize; i++)
                {
                    if (info.TryGetOutputIndex(index.Z, i, j, out int outputIndex))
                    {
                        float dP = dL * filter[info.FilterIndex(i, j, index.Y)];
                        Atomic.Add(ref output[outputIndex + outputOffset], dP);
                    }
                }
            }
        }
        /// <summary>
        /// Backpropagates through the layer without updating any of the filter weights. Called when learning rate is zero.
        /// </summary>
        protected override void BackwardsNoUpdate(int batchSize)
        {
            _buffers.OutGradient.SubView(0, batchSize * _inputShape.Volume).MemSetToZero();

            Index3D index = new(batchSize, _outputShape.Dimensions * _inputShape.Dimensions, _inputShape.Area);
            BackwardsOutGradientAction(index, _buffers.InGradient, _buffers.OutGradient, _filters.WeightsGPU<float>(), Info);

            Synchronize();

            _filters.DecrementLiveWeights();
        }


        /// <summary>
        /// Perform standard backpropagation through the layer, updating it's weights. Called when learning rate is greater than 0.
        /// </summary>
        /// <param name="learningRate">The overall learning rate for the layer updates, corrected for the influence of bias in the first and second moments.</param>
        /// <param name="firstMomentDecay">The exponential decay rate for the first moment.</param>
        /// <param name="secondMomentDecay">The exponential decay rate for the second moment.</param>
        protected override void BackwardsUpdate(int batchSize)
        {

            _buffers.OutGradient.SubView(0, batchSize * _inputShape.Volume).MemSetToZero();

            Index3D index = new(batchSize, _outputShape.Dimensions * _inputShape.Dimensions, _inputShape.Area);
            BackwardsOutGradientAction(index, _buffers.InGradient, _buffers.OutGradient, _filters.WeightsGPU<float>(), Info);
            BackwardsFilterAction(index, _buffers.InGradient, _inputCopy.GetArrayView<float>(), _filters.GradientGPU<float>(), Info);

            Synchronize();

            _inputCopy.DecrementLiveCount();

            _filters.DecrementLiveGradient();
            _filters.DecrementLiveWeights();
            _filters.UpdateWeights(_adamHyperParameters);

        }
    }
}