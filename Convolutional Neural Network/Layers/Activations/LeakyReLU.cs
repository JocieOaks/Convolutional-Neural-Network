using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.Layers.Activations
{
    /// <summary>
    /// The <see cref="LeakyReLU"/> class is an activation <see cref="Layer"/> that multiplies every element in a <see cref="Tensor"/> that is
    /// less than 0 by a value less than 1, in order to add non-linearity to the <see cref="Network"/>.
    /// </summary>
    [Serializable]
    public class LeakyReLU : Layer
    {
        private readonly float _negativeScaling;
        private static readonly Action<Index1D, ArrayView<int>, ArrayView<float>, float> s_backwardsAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<float>, float>(BackwardsKernel);
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<int>, float> s_forwardAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, float>(ForwardReLUKernel);
        private ArrayView<int> _deviceZeroed;
        /// <summary>
        /// Initializes a new instance of the <see cref="LeakyReLU"/> class.
        /// </summary>
        /// <param name="negativeScaling">Scaler to multiply <see cref="Tensor"/> values that are less than 0 by.</param>
        public LeakyReLU(float negativeScaling) : base(1, 1)
        {
            _negativeScaling = negativeScaling > 0 ? negativeScaling : 0.2f;
        }

        /// <inheritdoc/>
        public override string Name => "Activation Layer";

        /// <inheritdoc />
        public override bool Reflexive => true;

        /// <inheritdoc/>
        public override void Backwards(int batchSize, bool update)
        {
            Index1D index = new(InputShape.Area * batchSize * InputShape.Dimensions);
            s_backwardsAction(index, _deviceZeroed, Views.Gradient, _negativeScaling);

            GPUManager.Accelerator.Synchronize();
        }

        /// <inheritdoc/>
        public override void Forward(int batchSize)
        {
            Index1D index = new(InputShape.Area * batchSize * InputShape.Dimensions);
            s_forwardAction(index, Views.Input, _deviceZeroed, _negativeScaling);
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

        private static void BackwardsKernel(Index1D index, ArrayView<int> zeroed, ArrayView<float> inGradient, float negativeScaling)
        {
            int byteIndex = index.X / 32;
            int bit = index.X - 32 * byteIndex;
            int mask = 1 << bit;
            if ((zeroed[byteIndex] & mask) == 0)
            {
                inGradient[index.X] = negativeScaling * inGradient[index.X];
            }
        }

        private static void ForwardReLUKernel(Index1D index, ArrayView<float> input, ArrayView<int> zeroed, float negativeScaling)
        {
            int byteIndex = index.X / 32;
            int bit = index.X - 32 * byteIndex;
            int mask = 1 << bit;
            if (input[index.X] < 0)
            {
                Atomic.And(ref zeroed[byteIndex], ~mask);
                input[index.X] = negativeScaling * input[index.X];
            }
            else
            {
                Atomic.Or(ref zeroed[byteIndex], mask);
            }
        }
    }
}