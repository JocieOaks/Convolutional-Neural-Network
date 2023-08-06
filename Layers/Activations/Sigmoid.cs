using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers.Activations
{
    public class Sigmoid : Layer, IStructuralLayer, IUnchangedLayer
    {
        private Vector _inputCopy;

        public override string Name => "Sigmoid Activation";

        [JsonConstructor] public Sigmoid() : base(1, 1) { }

        public override void Forward(int batchSize)
        {
            Index1D index = new(batchSize * _inputShape.Volume);
            GPUManager.CopyAction(index, _buffers.Input, _inputCopy.GetArrayViewEmpty<float>());
            ForwardAction(index, _buffers.Input);

            Synchronize();

            _inputCopy.DecrementLiveCount();
        }

        public override void Backwards(int batchSize, bool update)
        {
            Index1D index = new(batchSize * _inputShape.Volume);
            BackwardsAction(index, _inputCopy.GetArrayView<float>(), _buffers.Gradient);
            Synchronize();

            _inputCopy.DecrementLiveCount();
        }

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

        public override void Reset()
        {
        }

        public override Shape Startup(Shape inputShape, IOBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            BaseStartup(inputShape, buffers, maxBatchSize);
            _inputCopy = new Vector(maxBatchSize * inputShape.Volume);
            return _outputShape;
        }
    }
}
