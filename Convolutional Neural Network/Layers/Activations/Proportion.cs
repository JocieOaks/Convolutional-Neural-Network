using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.Layers.Activations
{
    /// <summary>
    /// The <see cref="Proportion"/> class is a <see cref="Layer"/> that scales the values of a batch so that they add up to 1.
    /// Primarily this is useful as the final layer for a multiclass classification discriminator, as it calculates
    /// the probability that each classification is the correct one (assuming classifications are mutually exclusive).
    /// Values should be greater than or equal to 0, otherwise results may be unusual.
    /// </summary>
    public class Proportion : Layer
    {
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int> s_backwardsAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(ProportionGradientKernel);
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int> s_forwardAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int>(ProportionKernel);
        private Vector _inputCopy;
        private Vector _sum;

        /// <summary>
        /// Initializes a new instance of the <see cref="Proportion"/> class.
        /// </summary>
        public Proportion() : base(1, 1) { }

        /// <inheritdoc />
        public override string Name => "Proportion Layer";

        /// <inheritdoc />
        public override bool Reflexive => true;

        /// <inheritdoc />
        public override void Backwards(int batchSize, bool update)
        {
            Index1D index = new(batchSize);
            s_backwardsAction(index, _inputCopy.GetArrayView(), Views.Gradient, _sum.GetArrayView(), InputShape.Volume);
            GPUManager.Accelerator.Synchronize();

            _inputCopy.Release();
            _sum.Release();
        }

        /// <inheritdoc />
        public override void Forward(int batchSize)
        {
            Index1D copyIndex = new(batchSize * InputShape.Volume);
            GPUManager.CopyAction(copyIndex, Views.Input, _inputCopy.GetArrayViewEmpty());
            GPUManager.Accelerator.Synchronize();

            Index1D index = new(batchSize);
            s_forwardAction(index, Views.Input, _sum.GetArrayView(), InputShape.Volume);
            GPUManager.Accelerator.Synchronize();

            _inputCopy.Release();
            _sum.Release();
        }

        /// <inheritdoc />
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int maxBatchSize)
        {
            if (Initialized)
                return OutputShape;
            Initialized = true;

            BaseStartup(inputShape, views);
            _inputCopy = new Vector(maxBatchSize * OutputShape.Volume);
            _sum = new Vector(maxBatchSize);

            return OutputShape;
        }

        private static void ProportionGradientKernel(Index1D index, ArrayView<float> output, ArrayView<float> gradient, ArrayView<float> sum, int length)
        {
            int offset = index * length;
            float gradientSum = 0;
            float sumSquared = sum[index] * sum[index];

            for (int i = 0; i < length; i++)
            {
                gradientSum += output[offset + i] * gradient[offset + i] / sumSquared;
            }

            for (int i = 0; i < length; i++)
            {
                gradient[offset + i] /= sum[index];
                gradient[offset + i] -= gradientSum;
            }
        }

        private static void ProportionKernel(Index1D index, ArrayView<float> input, ArrayView<float> sum, int length)
        {
            int offset = index * length;
            sum[index] = Utility.ASYMPTOTE_ERROR_CORRECTION;
            for (int i = 0; i < length; i++)
            {
                sum[index] += input[offset + i];
            }

            for (int i = 0; i < length; i++)
            {
                input[offset + i] /= sum[index];
            }
        }
    }
}
