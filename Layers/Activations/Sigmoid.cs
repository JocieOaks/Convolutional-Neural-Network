using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System.Text.Json.Serialization;

namespace ConvolutionalNeuralNetwork.Layers.Activations
{
    public class Sigmoid : Layer
    {
        private Vector _inputCopy;

        public override string Name => "Sigmoid Activation";

        [JsonConstructor] public Sigmoid() : base(1, 1) { }

        public override void Forward(int batchSize)
        {
            Index1D index = new(batchSize * InputShape.Volume);
            GPUManager.CopyAction(index, Buffers.Input, _inputCopy.GetArrayViewEmpty());
            ForwardAction(index, Buffers.Input);

            Synchronize();

            _inputCopy.Release();
        }

        public override void Backwards(int batchSize, bool update)
        {
            Index1D index = new(batchSize * InputShape.Volume);
            BackwardsAction(index, _inputCopy.GetArrayView(), Buffers.Gradient);
            Synchronize();

            _inputCopy.Release();
        }

        /// <inheritdoc />
        [JsonIgnore] public override bool Reflexive => true;

        private static readonly Action<Index1D, ArrayView<float>> ForwardAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(SigmoidKernel);
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> BackwardsAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SigmoidGradientKernel);

        private static void SigmoidKernel(Index1D index, ArrayView<float> input)
        {
            input[index.X] = 1 / (1 + XMath.Exp(-input[index.X]));
        }

        private static void SigmoidGradientKernel(Index1D index, ArrayView<float> input, ArrayView<float> gradient)
        {
            float exp = XMath.Exp(-input[index.X]);
            float sig = 1 / (1 + exp);
            gradient[index.X] = exp * sig * sig * gradient[index.X];
        }

        public override TensorShape Startup(TensorShape inputShape, PairedBuffers buffers, int maxBatchSize)
        {
            if (Ready)
                return OutputShape;
            Ready = true;

            BaseStartup(inputShape, buffers);
            _inputCopy = new Vector(maxBatchSize * inputShape.Volume);
            return OutputShape;
        }
    }
}
