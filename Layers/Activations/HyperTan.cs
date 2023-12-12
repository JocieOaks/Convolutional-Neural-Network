using System.Text.Json.Serialization;
using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.Layers.Activations
{
    public class HyperTan : Layer, IReflexiveLayer
    {
        private Vector _outputCopy;

        public override string Name => "Hyperbolic Tangent Activation";

        [JsonConstructor] public HyperTan() : base(1, 1) { }

        public override void Backwards(int batchSize, bool update)
        {
            Index1D index = new(batchSize * _inputShape.Volume);
            BackwardsAction(index, _outputCopy.GetArrayView(), _buffers.Gradient);
            Synchronize();

            _outputCopy.Release();
        }

        public override void Forward(int batchSize)
        {
            Index1D index = new(batchSize * _inputShape.Volume);
            ForwardAction(index, _buffers.Input);
            Synchronize();

            GPUManager.CopyAction(index, _buffers.Input, _outputCopy.GetArrayViewEmpty());
            Synchronize();

            _outputCopy.Release();
        }

        private static readonly Action<Index1D, ArrayView<float>> ForwardAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(HyperTanKernel);

        private static void HyperTanKernel(Index1D index, ArrayView<float> input)
        {
            input[index.X] = XMath.Tanh(input[index.X]);
        }

        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> BackwardsAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(HyperTanGradientKernel);

        private static void HyperTanGradientKernel(Index1D index, ArrayView<float> output, ArrayView<float> gradient)
        {
            gradient[index.X] = gradient[index.X] * (1 - XMath.Pow(output[index.X], 2));
        }

        public override TensorShape Startup(TensorShape inputShape, PairedBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            BaseStartup(inputShape, buffers);
            _outputCopy = new Vector(maxBatchSize * _outputShape.Volume);

            return _outputShape;
        }
    }
}
