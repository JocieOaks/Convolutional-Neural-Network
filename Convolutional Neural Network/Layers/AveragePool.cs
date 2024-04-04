using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="AveragePool"/> class is a <see cref="Layer"/> that downscales the input <see cref="Tensor"/> by averaging a square of entries.
    /// </summary>
    public class AveragePool : Layer
    {
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo> s_backwardsAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo>(BackwardsKernel);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo> s_forwardAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo>(ForwardPoolKernel);

        /// <summary>
        /// Initializes a new instance of the <see cref="AveragePool"/> class.
        /// </summary>
        /// <param name="filterSize">The width and height of a block of pixels to be averaged.</param>
        public AveragePool(int filterSize) : base(filterSize, filterSize)
        {
        }

        /// <inheritdoc/>
        public override string Name => "Average Pool Layer";

        /// <inheritdoc/>
        public override void Backwards(int batchSize, bool update)
        {

            Index3D index = new(OutputShape.Area, InputShape.Dimensions, batchSize);
            s_backwardsAction(index, Views.InGradient, Views.OutGradient, LayerInfo);

            GPUManager.Accelerator.Synchronize();
        }

        /// <inheritdoc/>
        public override void Forward(int batchSize)
        {

            Index3D index = new(OutputShape.Area, InputShape.Dimensions, batchSize);
            s_forwardAction(index, Views.Input, Views.Output, LayerInfo);

            GPUManager.Accelerator.Synchronize();
        }

        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int maxBatchSize)
        {
            if (Initialized)
                return OutputShape;
            Initialized = true;

            BaseStartup(inputShape, views);
            
            return OutputShape;
        }

        /// <summary>
        /// An ILGPU kernel to calculate the gradients for backpropagating the previous layer.
        /// </summary>
        private static void BackwardsKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, LayerInfo info)
        {
            (int inputOffset, int outputOffset) = info.GetOffset(index.Y, index.Z);

            float dL = inGradient[index.X + outputOffset] * info.InverseFilterArea;

            for (int j = 0; j < info.FilterSize; j++)
            {
                for (int i = 0; i < info.FilterSize; i++)
                {
                    if (info.TryGetExpansionIndex(index.X, i, j, out int inputIndex))
                    {
                        outGradient[inputIndex + inputOffset] = dL;
                    }
                }
            }
        }

        /// <summary>
        /// An ILGPU kernel to downscale the input <see cref="Tensor"/>.
        /// </summary>
        private static void ForwardPoolKernel(Index3D index, ArrayView<float> input, ArrayView<float> pooled, LayerInfo info)
        {
            (int inputOffset, int outputOffset) = info.GetOffset(index.Y, index.Z);

            float sum = 0;
            for (int j = 0; j < info.FilterSize; j++)
            {
                for (int i = 0; i < info.FilterSize; i++)
                {
                    if (info.TryGetExpansionIndex(index.X, i, j, out int inputIndex))
                        sum += input[inputIndex + inputOffset];
                }
            }

            pooled[index.X + outputOffset] = sum * info.InverseFilterArea;
        }
    }
}