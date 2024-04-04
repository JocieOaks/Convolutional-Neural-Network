using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.Layers.Activations
{
    /// <summary>
    /// The <see cref="Sigmoid"/> class is an activation <see cref="Layer"/> that runs every element of the
    /// input <see cref="Tensor"/> through the sigmoid function to add non-linearity to the <see cref="Network"/>
    /// </summary>
    public class Sigmoid : Layer
    {
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> s_backwardsAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SigmoidGradientKernel);
        private static readonly Action<Index1D, ArrayView<float>> s_forwardAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(SigmoidKernel);
        private Vector _inputCopy;

        /// <summary>
        /// Initializes a new instance of the <see cref="Sigmoid"/> class.
        /// </summary>
        public Sigmoid() : base(1, 1) { }

        /// <inheritdoc />
        public override string Name => "Sigmoid Activation";

        /// <inheritdoc />
        public override bool Reflexive => true;

        /// <inheritdoc />
        public override void Backwards(int batchSize, bool update)
        {
            Index1D index = new(batchSize * InputShape.Volume);
            s_backwardsAction(index, _inputCopy.GetArrayView(), Views.Gradient);
            GPUManager.Accelerator.Synchronize();

            _inputCopy.Release();
        }

        /// <inheritdoc />
        public override void Forward(int batchSize)
        {
            Index1D index = new(batchSize * InputShape.Volume);
            GPUManager.CopyAction(index, Views.Input, _inputCopy.GetArrayViewEmpty());
            s_forwardAction(index, Views.Input);

            GPUManager.Accelerator.Synchronize();

            _inputCopy.Release();
        }

        /// <inheritdoc />
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int maxBatchSize)
        {
            if (Initialized)
                return OutputShape;
            Initialized = true;

            BaseStartup(inputShape, views);
            _inputCopy = new Vector(maxBatchSize * inputShape.Volume);
            return OutputShape;
        }

        private static void SigmoidGradientKernel(Index1D index, ArrayView<float> input, ArrayView<float> gradient)
        {
            float exp = XMath.Exp(-input[index.X]);
            float sig = 1 / (1 + exp);
            gradient[index.X] = exp * sig * sig * gradient[index.X];
        }

        private static void SigmoidKernel(Index1D index, ArrayView<float> input)
        {
            input[index.X] = 1 / (1 + XMath.Exp(-input[index.X]));
        }
    }
}
