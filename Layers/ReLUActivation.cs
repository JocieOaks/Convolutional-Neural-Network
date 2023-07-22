using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="ReLUActivation"/> class is a <see cref="Layer"/> is an activation to add non-linearity to the <see cref="Network"/>.
    /// </summary>
    [Serializable]
    public class ReLUActivation : Layer, ISecondaryLayer
    {
        private static readonly Action<Index1D, ArrayView<int>, ArrayView<float>, ArrayView<float>> s_backwardsAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<float>, ArrayView<float>>(BackwardsKernel);
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>> s_forwardAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>>(ForwardKernel);
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
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            Index1D index = new(_inputShape.Area * _batchSize * _inputDimensions);
            s_backwardsAction(index, _deviceZeroed, _buffers.InGradient, _buffers.OutGradient);

            Synchronize();
        }
        /// <inheritdoc/>
        public override void Forward()
        {
            Index1D index = new(_inputShape.Area * _batchSize * _inputDimensions);
            s_forwardAction(index, _buffers.Input, _deviceZeroed, _buffers.Output);
            Synchronize();
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override Shape Startup(Shape inputShapes, IOBuffers buffers, int batchSize)
        {
            BaseStartup(inputShapes, buffers, batchSize);

            int zeroArea = inputShapes.Area / 32 + (inputShapes.Area % 32 > 0 ? 1 : 0);
            zeroArea *= _inputDimensions;
            zeroArea *= _batchSize;

            _deviceZeroed = GPU.GPUManager.Accelerator.Allocate1D<int>(zeroArea).View;
            return _outputShape;
        }

        private static void BackwardsKernel(Index1D index, ArrayView<int> zeroed, ArrayView<float> inGradient, ArrayView<float> outGradient)
        {
            int byteIndex = (index.X) / 32;
            int bit = index.X - 32 * byteIndex;
            int mask = (1 << bit);
            if ((zeroed[byteIndex] & mask) != 0)
            {
                outGradient[index.X] = inGradient[index.X];
            }
            else
            {
                outGradient[index.X] = NEGATIVESCALING * inGradient[index.X];
            }
        }

        private static void ForwardKernel(Index1D index, ArrayView<float> input, ArrayView<int> zeroed, ArrayView<float> output)
        {
            int byteIndex = (index.X) / 32;
            int bit = index.X - 32 * byteIndex;
            int mask = (1 << bit);
            if(input[index.X] < 0)
            {
                Atomic.And(ref zeroed[byteIndex], ~mask);
                output[index.X] = NEGATIVESCALING * input[index.X];
            }
            else
            {
                Atomic.Or(ref zeroed[byteIndex], mask);
                output[index.X] = input[index.X];
            }
        }
    }
}