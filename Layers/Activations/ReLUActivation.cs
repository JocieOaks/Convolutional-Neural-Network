using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Activations
{
    /// <summary>
    /// The <see cref="ReLUActivation"/> class is a <see cref="Layer"/> is an activation to add non-linearity to the <see cref="Network"/>.
    /// </summary>
    [Serializable]
    public class ReLUActivation : Layer, ISecondaryLayer, IUnchangedLayer
    {
        private static readonly Action<Index1D, ArrayView<int>, ArrayView<float>> s_backwardsAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<float>>(BackwardsKernel);
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<int>> s_forwardAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>>(ForwardKernel);
        private ArrayView<int> _deviceZeroed;

        private const float NEGATIVESCALING = 0.2f;

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
            Index1D index = new(_inputShape.Area * batchSize * _inputShape.Dimensions);
            s_backwardsAction(index, _deviceZeroed, _buffers.Gradient);

            Synchronize();
        }
        /// <inheritdoc/>
        public override void Forward(int batchSize)
        {
            Index1D index = new(_inputShape.Area * batchSize * _inputShape.Dimensions);
            s_forwardAction(index, _buffers.Input, _deviceZeroed);
            Synchronize();
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override Shape Startup(Shape inputShapes, IOBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            BaseStartup(inputShapes, buffers);

            int zeroArea = inputShapes.Area / 32 + (inputShapes.Area % 32 > 0 ? 1 : 0);
            zeroArea *= _inputShape.Dimensions;
            zeroArea *= maxBatchSize;

            _deviceZeroed = GPU.GPUManager.Accelerator.Allocate1D<int>(zeroArea).View;
            return _outputShape;
        }

        private static void BackwardsKernel(Index1D index, ArrayView<int> zeroed, ArrayView<float> inGradient)
        {
            int byteIndex = index.X / 32;
            int bit = index.X - 32 * byteIndex;
            int mask = 1 << bit;
            if ((zeroed[byteIndex] & mask) == 0)
            {
                inGradient[index.X] = NEGATIVESCALING * inGradient[index.X];
            }
        }

        private static void ForwardKernel(Index1D index, ArrayView<float> input, ArrayView<int> zeroed)
        {
            int byteIndex = index.X / 32;
            int bit = index.X - 32 * byteIndex;
            int mask = 1 << bit;
            if (input[index.X] < 0)
            {
                Atomic.And(ref zeroed[byteIndex], ~mask);
                input[index.X] = NEGATIVESCALING * input[index.X];
            }
            else
            {
                Atomic.Or(ref zeroed[byteIndex], mask);
            }
        }
    }
}