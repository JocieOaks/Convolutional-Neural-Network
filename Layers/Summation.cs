using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.Layers
{

    /// <summary>
    /// The <see cref="Summation"/> class is a <see cref="Layer"/> that sums a <see cref="Tensor"/> across multiple dimensions, reducing the number of
    /// dimensions in the <see cref="Tensor"/>.
    /// Note: When summing <see cref="Tensor"/>s that have been batch normalized to have a mean of 0, the mean will remain the same, but the standard deviation
    /// will change.
    /// </summary>
    [Serializable]
    public class Summation : Layer
    {
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, TensorShape, int> s_backwardsAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, TensorShape, int>(SummationGradientKernel);

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, TensorShape, int> s_forwardAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, TensorShape, int>(SummationKernel);

        public Summation(int outputDimensions) : base(1, 1)
        {
            _outputShape = new TensorShape(0, 0, outputDimensions);
        }
        /// <inheritdoc/>
        public override string Name => "Summation Layer";
        /// <inheritdoc/>
        public override void Backwards(int batchSize, bool update)
        {
            Index3D index = new(batchSize, _inputShape.Dimensions, _inputShape.Area);
            s_backwardsAction(index, _buffers.Input, _buffers.Output, _inputShape, _outputShape.Dimensions);

            Synchronize();
        }

        /// <inheritdoc/>
        public override void Forward(int batchSize)
        {
            _buffers.Output.SubView(0, batchSize * _outputShape.Dimensions * _inputShape.Area).MemSetToZero();

            Index3D index = new(batchSize, _inputShape.Dimensions, _inputShape.Area);
            s_forwardAction(index, _buffers.Input, _buffers.Output, _inputShape, _outputShape.Dimensions);

            Synchronize();
        }

        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShapes, PairedBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            BaseStartup(inputShapes, buffers, _outputShape.Dimensions);

            return _outputShape;
        }

        private static void SummationGradientKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, TensorShape shape, int outputDimensions)
        {
            int outGradientIndex = (index.X * shape.Dimensions + index.Y) * shape.Area + index.Z;
            int inGradientIndex = (index.X * outputDimensions + index.Y % outputDimensions) * shape.Area + index.Z;

            outGradient[outGradientIndex] = inGradient[inGradientIndex];
        }

        private static void SummationKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, TensorShape shape, int outputDimensions)
        {
            int inputIndex = (index.X * shape.Dimensions + index.Y) * shape.Area + index.Z;
            int outputIndex = (index.X * outputDimensions + index.Y % outputDimensions) * shape.Area + index.Z;

            Atomic.Add(ref output[outputIndex], input[inputIndex]);
        }
    }
}
