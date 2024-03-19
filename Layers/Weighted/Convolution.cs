using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.Layers.Weighted
{
    /// <summary>
    /// The <see cref="Convolution"/> class is a <see cref="Layer"/> that performs a 2D Convolution,
    /// by passing a 2D filter over a <see cref="Tensor"/> and summing the products to output a new <see cref="Tensor"/>.
    /// </summary>
    [Serializable]
    public class Convolution : WeightedLayer
    {
        private static readonly Action<KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> s_backwardsFilterAction = GPUManager.Accelerator.LoadStreamKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(ConvFilterKernel);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> s_backwardsOutGradientAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(ConvGradientKernel);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> s_forwardAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(ConvKernel);
        private readonly int _outputDimensions;

        /// <summary>
        /// Initializes a new instance of the <see cref="Convolution"/> class.
        /// </summary>
        /// <param name="filterSize">The width and height of a filter.</param>
        /// <param name="stride">The amount of movement over the image for each filter pass.</param>
        /// <param name="outputDimensions">A factor relating the number of input layers to the number of output layers.
        /// Must be positive. To reduce the number of output dimensions, use a <see cref="Summation"/> layer afterwards.</param>
        /// <param name="weights">The initial weights for the <see cref="Layer"/>'s filters.</param>
        /// <param name="bias">The initial weights for the <see cref="Layer"/>'s bias.</param>
        public Convolution(int filterSize, int stride, int outputDimensions, Weights weights, Weights bias) : base (filterSize, stride, weights, bias)
        {
            _outputDimensions = outputDimensions;
        }

        /// <inheritdoc/>
        public override string Name => "Convolutional Layer";

        private LayerInfo Info => LayerInfo;
        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int maxBatchSize)
        {
            if (Initialized)
                return OutputShape;
            Initialized = true;

            if (Weights == null)
            {
                BaseStartup(inputShape, views, _outputDimensions);
            }
            else
            {
                BaseStartup(inputShape, views, Weights.Length / FilterSize / FilterSize / inputShape.Dimensions);
            }

            InputCopy = new Vector(InputShape.Dimensions * maxBatchSize * inputShape.Area);
            return OutputShape;
        }

        /// <summary>
        /// Back-propagates through the layer without updating any of the filter weights. Called when learning rate is zero.
        /// </summary>
        protected override void BackwardsNoUpdate(int batchSize)
        {
            Views.OutGradient.SubView(0, batchSize * InputShape.Volume).MemSetToZero();

            Index3D gradientIndex = new(InputShape.Volume, OutputShape.Dimensions, batchSize);
            s_backwardsOutGradientAction(gradientIndex, Views.InGradient, Weights.WeightsView(), Views.OutGradient, Info);
        }

        /// <summary>
        /// Perform standard back-propagation through the layer, updating it's weights. Called when learning rate is greater than 0.
        /// </summary>
        /// <param name="batchSize"></param>
        /// 
        /// 
        /// 
        protected override void BackwardsUpdate(int batchSize)
        {
            Views.OutGradient.SubView(0, batchSize * InputShape.Volume).MemSetToZero();

            Index3D gradientIndex = new(InputShape.Volume, OutputShape.Dimensions, batchSize);

            s_backwardsOutGradientAction(gradientIndex, Views.InGradient, Weights.WeightsView(), Views.OutGradient, Info);
            KernelConfig config = new(new Index3D(Info.FilterArea, InputShape.Dimensions, batchSize), new Index3D(OutputShape.Dimensions, 1, 1));
            s_backwardsFilterAction(config, Views.InGradient, InputCopy.GetArrayView(), Weights.GradientView(), Info);
        }

        /// <inheritdoc/>
        protected override void ForwardChild(int batchSize)
        {
            Index1D copyIndex = new(InputShape.Volume * batchSize);
            GPUManager.CopyAction(copyIndex, Views.Input, InputCopy.GetArrayViewEmpty());

            Views.Output.SubView(0, batchSize * OutputShape.Volume).MemSetToZero();
            Index3D index = new(OutputShape.Volume, InputShape.Dimensions, batchSize);
            s_forwardAction(index, Views.Input, Views.Output, Weights.WeightsView(), Info);
        }

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