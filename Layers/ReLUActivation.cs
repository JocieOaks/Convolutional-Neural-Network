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
        private MemoryBuffer1D<int, Stride1D.Dense>[,] _deviceZeroed;

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
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index1D index = new(Infos(i).Area * 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    s_backwardsAction(index, _deviceZeroed[i, j].View, _buffers.InGradientsFloat[i, j], _buffers.OutGradientsFloat[i, j]);
                }
            }
            Synchronize();
        }
        /// <inheritdoc/>
        public override void Forward()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index1D index = new(3 * Infos(i).Area);

                for (int j = 0; j < _batchSize; j++)
                {
                    s_forwardAction(index, _buffers.InputsFloat[i, j], _deviceZeroed[i, j].View, _buffers.OutputsFloat[i, j]);
                }
            }
            Synchronize();
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override FeatureMap[,] Startup(FeatureMap[,] inputs, IOBuffers buffers)
        {
            BaseStartup(inputs, buffers);

            _deviceZeroed = new MemoryBuffer1D<int, Stride1D.Dense>[_inputDimensions, _batchSize];

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceZeroed[i, j] = GPU.GPUManager.Accelerator.Allocate1D<int>(inputs[i, j].FloatLength / 32 + (inputs[i,j].FloatLength % 32 > 0 ? 1 : 0));
                }
            }
            return _outputs;
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

        /// <summary>
        /// Gets the <see cref="StaticLayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="StaticLayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="StaticLayerInfo"/> corresponding to an input dimension.</returns>
        private StaticLayerInfo Infos(int index)
        {
            return (StaticLayerInfo)_layerInfos[index];
        }
    }
}