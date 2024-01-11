using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Activations
{
    /// <summary>
    /// The <see cref="ReLUActivation"/> class is a <see cref="Layer"/> is an activation to add non-linearity to the <see cref="Network"/>.
    /// </summary>
    [Serializable]
    public class ReLUActivation : Layer
    {
        private static readonly Action<Index1D, ArrayView<int>, ArrayView<float>> s_backwardsAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<float>>(BackwardsKernel);
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<int>> s_forwardAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>>(ForwardReLUKernel);
        private ArrayView<int> _deviceZeroed;

        private const float NEGATIVE_SCALING = 0.2f;

        /// <summary>
        /// Initializes a new instance of the <see cref="ReLUActivation"/> class.
        /// </summary>
        [JsonConstructor]
        public ReLUActivation() : base(1, 1)
        {
        }

        /// <inheritdoc/>
        public override string Name => "Activation Layer";
        /// <inheritdoc/>
        public override void Backwards(int batchSize, bool update)
        {
            Index1D index = new(InputShape.Area * batchSize * InputShape.Dimensions);
            s_backwardsAction(index, _deviceZeroed, Buffers.Gradient);

            Synchronize();
        }
        /// <inheritdoc/>
        public override void Forward(int batchSize)
        {
            Index1D index = new(InputShape.Area * batchSize * InputShape.Dimensions);
            s_forwardAction(index, Buffers.Input, _deviceZeroed);
            Synchronize();
        }

        /// <inheritdoc />
        [JsonIgnore] public override bool Reflexive => true;

        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShapes, PairedBuffers buffers, int maxBatchSize)
        {
            if (Ready)
                return OutputShape;
            Ready = true;

            BaseStartup(inputShapes, buffers);

            int zeroArea = inputShapes.Area / 32 + (inputShapes.Area % 32 > 0 ? 1 : 0);
            zeroArea *= InputShape.Dimensions;
            zeroArea *= maxBatchSize;

            _deviceZeroed = GPU.GPUManager.Accelerator.Allocate1D<int>(zeroArea).View;
            return OutputShape;
        }

        private static void BackwardsKernel(Index1D index, ArrayView<int> zeroed, ArrayView<float> inGradient)
        {
            int byteIndex = index.X / 32;
            int bit = index.X - 32 * byteIndex;
            int mask = 1 << bit;
            if ((zeroed[byteIndex] & mask) == 0)
            {
                inGradient[index.X] = NEGATIVE_SCALING * inGradient[index.X];
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
                input[index.X] = NEGATIVE_SCALING * input[index.X];
            }
            else
            {
                Atomic.Or(ref zeroed[byteIndex], mask);
            }
        }
    }
}