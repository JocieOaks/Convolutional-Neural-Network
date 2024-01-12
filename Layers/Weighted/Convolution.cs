using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Weighted
{
    /// <summary>
    /// The <see cref="Convolution"/> class is a <see cref="Layer"/> that performs the titular convolutions of a convolutional
    /// neural network, by passing <see cref="FeatureMap"/>s through a variety of filters.
    /// </summary>
    [Serializable]
    public class Convolution : WeightedLayer
    {
        private readonly int _outputDimensions;

        public Convolution(int filterSize, int stride, int outputDimensions, Weights weights, Weights bias) : base (filterSize, stride, weights, bias)
        {
            _outputDimensions = outputDimensions;
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor]
        private Convolution() : base()
        {
        }

        public static Action<KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> BackwardsFilterAction { get; } = GPUManager.Accelerator.LoadStreamKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(ConvFilterKernel);

        public static Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> BackwardsOutGradientAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(ConvGradientKernel);

        public static Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> ForwardAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(ConvKernel);

        /// <inheritdoc/>
        public override string Name => "Convolutional Layer";

        private LayerInfo Info => (LayerInfo)LayerInfo;

        protected override int WeightLength => FilterSize * FilterSize * OutputShape.Dimensions * InputShape.Dimensions;

        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int maxBatchSize)
        {
            if (Initialized)
                return OutputShape;
            Initialized = true;

            if (_weights == null)
            {
                BaseStartup(inputShape, views, _outputDimensions);
            }
            else
            {
                BaseStartup(inputShape, views, _weights.Length / FilterSize / FilterSize / inputShape.Dimensions);
            }

            _inputCopy = new Vector(InputShape.Dimensions * maxBatchSize * inputShape.Area);
            return OutputShape;
        }

        /// <summary>
        /// Backpropagates through the layer without updating any of the filter weights. Called when learning rate is zero.
        /// </summary>
        protected override void BackwardsNoUpdate(int batchSize)
        {
            Views.OutGradient.SubView(0, batchSize * InputShape.Volume).MemSetToZero();

            Index3D gradientIndex = new(InputShape.Volume, OutputShape.Dimensions, batchSize);
            BackwardsOutGradientAction(gradientIndex, Views.InGradient, _weights.WeightsView(), Views.OutGradient, Info);
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
            Views.OutGradient.SubView(0, batchSize * InputShape.Volume).MemSetToZero();

            Index3D gradientIndex = new(InputShape.Volume, OutputShape.Dimensions, batchSize);

            var stream = GPUManager.Accelerator.DefaultStream;
            BackwardsOutGradientAction(gradientIndex, Views.InGradient, _weights.WeightsView(), Views.OutGradient, Info);
            KernelConfig config = new(new Index3D(Info.FilterArea, InputShape.Dimensions, batchSize), new Index3D(OutputShape.Dimensions, 1, 1));
            BackwardsFilterAction(config, Views.InGradient, _inputCopy.GetArrayView(), _weights.GradientView(), Info);
        }

        /// <inheritdoc/>
        protected override void ForwardChild(int batchSize)
        {
            Index1D copyIndex = new(InputShape.Volume * batchSize);
            GPUManager.CopyAction(copyIndex, Views.Input, _inputCopy.GetArrayViewEmpty());

            Views.Output.SubView(0, batchSize * OutputShape.Volume).MemSetToZero();
            Index3D index = new(OutputShape.Volume, InputShape.Dimensions, batchSize);
            ForwardAction(index, Views.Input, Views.Output, _weights.WeightsView(), Info);
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
        private static void ConvFilterKernel(ArrayView<float> inGradient, ArrayView<float> input, ArrayView<float> filterGradient, LayerInfo info)
        {

            int outputOffset = (Grid.IdxZ * info.ContractionDimensions + Group.IdxX) * info.ContractionArea;
            int inputOffset = (Grid.IdxZ * info.ExpansionDimensions + Grid.IdxY) * info.ExpansionArea;
            int dimension = Grid.IdxY * info.ContractionDimensions + Group.IdxX;
            float dK = 0;

            int x = Grid.IdxX % info.FilterSize;
            int y = Grid.IdxX / info.FilterSize;

            for (int i = 0; i < info.ContractionArea; i++)
            {
                if (info.TryGetExpansionIndex(i, x, y, out int inputIndex))
                {
                    float dL = inGradient[outputOffset + i];
                    dK += dL * input[inputIndex + inputOffset];
                }
            }

            int filterIndex = info.FilterIndex(x, y, dimension);
            Atomic.Add(ref filterGradient[filterIndex], dK);
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
        private static void ConvGradientKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> filter, ArrayView<float> outGradient, LayerInfo info)
        {
            info.DeconstructExpansion(index, out int mapIndex, out int inGradientOffset, out int outGradientIndex, out int dimension);

            (int x, int y) = info.GetExpansionCoordinates(mapIndex);

            int minX = x - info.FilterSize + info.Padding + 1;
            int minY = y - info.FilterSize + info.Padding + 1;
            int maxX = x + info.Padding + 1;
            int maxY = y + info.Padding + 1;

            int shiftX = minX % info.Stride;
            int shiftY = minY % info.Stride;

            shiftX -= XMath.Clamp(shiftX, 0, 1) * info.Stride;
            shiftY -= XMath.Clamp(shiftY, 0, 1) * info.Stride;

            int x0 = XMath.Max(0, maxX - info.ExpansionWidth);
            int x1 = XMath.Min(info.FilterSize + shiftX, info.FilterSize + minX);
            int y0 = XMath.Max(0, maxY - info.ExpansionLength);
            int y1 = XMath.Min(info.FilterSize + shiftY, info.FilterSize + minY);

            float sum = 0;
            
            for (int j = y1 - 1; j >= y0; j -= info.Stride)
            {
                for (int i = x1 - 1; i >= x0; i -= info.Stride)
                {
                    int inGradientIndex = info.GetContractionIndex(mapIndex, i, j);
                    int filterIndex = info.FilterIndex(i, j, dimension);
                    sum += inGradient[inGradientIndex + inGradientOffset] * filter[filterIndex];
                }
            }
            
            Atomic.Add(ref outGradient[outGradientIndex], sum);
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
        private static void ConvKernel(Index3D index, ArrayView<float> input, ArrayView<float> convoluted, ArrayView<float> filter, LayerInfo info)
        {
            info.DeconstructContraction(index, out int mapIndex, out int inputOffset, out int outputIndex, out int dimension);
            
            float sum = 0;

            (int x, int y) = info.GetContractionCoordinates(mapIndex);

            int minX = x * info.Stride - info.Padding;
            int minY = y * info.Stride - info.Padding;

            int x0 = XMath.Max(0, -minX);
            int x1 = XMath.Min(info.FilterSize, info.ExpansionWidth - minX);
            int y0 = XMath.Max(0, -minY);
            int y1 = XMath.Min(info.FilterSize, info.ExpansionLength - minY);

            for (int j = y0; j < y1; j++)
            {
                for (int i = x0; i < x1; i++)
                {
                    int inputIndex = info.GetExpansionIndex(mapIndex, i, j);
                    sum += filter[info.FilterIndex(i, j, dimension)] * input[inputIndex + inputOffset];
                }
            }

            Atomic.Add(ref convoluted[outputIndex], sum);
        }
    }
}