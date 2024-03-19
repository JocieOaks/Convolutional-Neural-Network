using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.Layers.Activations
{
    /// <summary>
    /// The <see cref="ReLU"/> class is an activation <see cref="Layer"/> that sets every element in a <see cref="Tensor"/> that is
    /// less than 0 to 0, in order to add non-linearity to the <see cref="Network"/>.
    /// </summary>
    [Serializable]
    public class ReLU : Layer
    {
        private static readonly Action<Index1D, ArrayView<int>, ArrayView<float>> s_backwardsAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<float>>(BackwardsKernel);
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<int>> s_forwardAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>>(ForwardReLUKernel);
        private ArrayView<int> _deviceZeroed;

        /// <summary>
        /// Initializes a new instance of the <see cref="ReLU"/> class.
        /// </summary>
        public ReLU() : base(1, 1)
        {
        }

        /// <inheritdoc/>
        public override string Name => "Activation Layer";

        /// <inheritdoc />
        public override bool Reflexive => true;

        /// <inheritdoc/>
        public override void Backwards(int batchSize, bool update)
        {
            Index1D index = new(InputShape.Area * batchSize * InputShape.Dimensions);
            s_backwardsAction(index, _deviceZeroed, Views.Gradient);

            GPUManager.Accelerator.Synchronize();
        }

        /// <inheritdoc/>
        public override void Forward(int batchSize)
        {
            Index1D index = new(InputShape.Area * batchSize * InputShape.Dimensions);
            s_forwardAction(index, Views.Input, _deviceZeroed);
            GPUManager.Accelerator.Synchronize();
        }

        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShapes, PairedGPUViews views, int maxBatchSize)
        {
            if (Initialized)
                return OutputShape;
            Initialized = true;

            BaseStartup(inputShapes, views);

            int zeroArea = inputShapes.Area / 32 + (inputShapes.Area % 32 > 0 ? 1 : 0);
            zeroArea *= InputShape.Dimensions;
            zeroArea *= maxBatchSize;

            _deviceZeroed = GPUManager.Accelerator.Allocate1D<int>(zeroArea).View;
            return OutputShape;
        }

        private static void BackwardsKernel(Index1D index, ArrayView<int> zeroed, ArrayView<float> inGradient)
        {
            int byteIndex = index.X / 32;
            int bit = index.X - 32 * byteIndex;
            int mask = 1 << bit;
            if ((zeroed[byteIndex] & mask) == 0)
            {
                inGradient[index.X] = 0;
            }
        }

        private static void ForwardReLUKernel(Index1D index, ArrayView<float> input, ArrayView<int> zeroed)
        {
            int byteIndex = index.X / 32;
            int bit = index.X - 32 * byteIndex;
            int mask = 1 << bit;
            if (input[index.X] < 0)
            {
                Atomic.And(ref zeroed[byteIndex], ~mask);
                input[index.X] = 0;
            }
            else
            {
                Atomic.Or(ref zeroed[byteIndex], mask);
            }
        }
    }
}