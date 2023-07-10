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
        private static readonly Action<Index1D, ArrayView<byte>, ArrayView<float>, ArrayView<float>> s_backwardsAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<float>, ArrayView<float>>(BackwardsKernal);
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<float>> s_forwardAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<float>>(ForwardKernal);
        private MemoryBuffer1D<byte, Stride1D.Dense>[,] _deviceZeroed;

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

            Utility.Accelerator.Synchronize();
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

            Utility.Accelerator.Synchronize();
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override FeatureMap[,] Startup(FeatureMap[,] inputs, IOBuffers buffers)
        {
            BaseStartup(inputs, buffers);

            _deviceZeroed = new MemoryBuffer1D<byte, Stride1D.Dense>[_inputDimensions, _batchSize];

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceZeroed[i, j] = Utility.Accelerator.Allocate1D<byte>(inputs[i, j].Area * 3 / 8 + 1);
                }
            }
            return _outputs;
        }

        private static void BackwardsKernal(Index1D index, ArrayView<byte> zeroed, ArrayView<float> inGradient, ArrayView<float> outGradient)
        {
            int byteIndex = (index.X) / 8;
            int bit = index.X % byteIndex;
            byte mask = (byte)(1 << bit);
            if ((zeroed[byteIndex] & mask) != 0)
            {
                outGradient[index.X] = inGradient[index.X];
            }
            else
            {
                outGradient[index.X] = 0.01f * inGradient[index.X];
            }
        }

        private static void ForwardKernal(Index1D index, ArrayView<float> input, ArrayView<byte> zeroed, ArrayView<float> output)
        {

            int byteIndex = (index.X) / 8;
            int bit = index.X % byteIndex;
            byte mask = (byte)(1 << bit);
            if(input[index.X] < 0)
            {
                zeroed[byteIndex] &= (byte)~mask;
                output[index.X] = 0.01f * input[index.X];
            }
            else
            {
                zeroed[byteIndex] |= mask;
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