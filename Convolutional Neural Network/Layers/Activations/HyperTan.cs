using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.Layers.Activations
{
    /// <summary>
    /// The <see cref="HyperTan"/> class is an activation <see cref="Layer"/> that runs every element of the
    /// input <see cref="Tensor"/> through the hyperbolic tangent function to add non-linearity to the <see cref="Network"/>
    /// </summary>
    public class HyperTan : Layer
    {
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> s_backwardsAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(HyperTanGradientKernel);
        private static readonly Action<Index1D, ArrayView<float>> s_forwardAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(HyperTanKernel);
        private Vector _outputCopy;

        /// <summary>
        /// Initializes a new instance of the <see cref="HyperTan"/> class.
        /// </summary>
        public HyperTan() : base(1, 1) { }

        /// <inheritdoc />
        public override string Name => "Hyperbolic Tangent Activation";
        /// <inheritdoc />
        public override bool Reflexive => true;

        /// <inheritdoc />
        public override void Backwards(int batchSize, bool update)
        {
            Index1D index = new(batchSize * InputShape.Volume);
            s_backwardsAction(index, _outputCopy.GetArrayView(), Views.Gradient);
            GPUManager.Accelerator.Synchronize();

            _outputCopy.Release();
        }

        /// <inheritdoc />
        public override void Forward(int batchSize)
        {
            Index1D index = new(batchSize * InputShape.Volume);
            s_forwardAction(index, Views.Input);
            GPUManager.Accelerator.Synchronize();

            GPUManager.CopyAction(index, Views.Input, _outputCopy.GetArrayViewEmpty());
            GPUManager.Accelerator.Synchronize();

            _outputCopy.Release();
        }

        /// <inheritdoc />
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int maxBatchSize)
        {
            if (Initialized)
                return OutputShape;
            Initialized = true;

            BaseStartup(inputShape, views);
            _outputCopy = new Vector(maxBatchSize * OutputShape.Volume);

            return OutputShape;
        }

        private static void HyperTanGradientKernel(Index1D index, ArrayView<float> output, ArrayView<float> gradient)
        {
            gradient[index.X] *= (1 - XMath.Pow(output[index.X], 2));
        }

        private static void HyperTanKernel(Index1D index, ArrayView<float> input)
        {
            input[index.X] = XMath.Tanh(input[index.X]);
        }
    }
}
