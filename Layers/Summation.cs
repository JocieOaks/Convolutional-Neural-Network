using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers
{

    /// <summary>
    /// The <see cref="Summation"/> class is a <see cref="Layer"/> that sums the <see cref="FeatureMap"/>s across multiple dimensions, reducing the number of
    /// dimensions in the <see cref="Network"/>.
    /// Note: When summing <see cref="FeatureMap"/>s that have been batch normalized to have a mean of 0, the mean will remain the same, but the standard deviation
    /// will change.
    /// </summary>
    [Serializable]
    public class Summation : Layer, IStructuralLayer
    {
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, Shape, int> s_backwardsAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, Shape, int>(BackwardsKernel);

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, Shape, int> s_forwardAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, Shape, int>(ForwardKernel);

        [JsonProperty]
        private Shape Output
        {
            get => _outputShape;
            set => _outputShape = value;
        }
        [JsonConstructor] public Summation() : base(1, 1) { }
        /// <inheritdoc/>
        public override string Name => "Summation Layer";
        /// <inheritdoc/>
        public override void Backwards(int batchSize)
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
        public override void Reset()
        {
        }

        /// <summary>
        /// Sets the number of dimensions for the output. The input dimensions will need to be a multiple of the output dimensions
        /// or vice versa. Overwrites <see cref="SetOutputDivisor(int)"/>.
        /// </summary>
        /// <param name="dimensions">The number of output dimensions.</param>
        public void SetOutputDimensions(int dimensions)
        {
            _outputShape = new Shape(0, 0, dimensions);
        }

        /// <inheritdoc/>
        public override Shape Startup(Shape inputShapes, IOBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            BaseStartup(inputShapes, buffers, _outputShape.Dimensions);

            return _outputShape;
        }

        private static void BackwardsKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, Shape shape, int outputDimensions)
        {
            int outGradientIndex = (index.X * shape.Dimensions + index.Y) * shape.Area + index.Z;
            int inGradientIndex = (index.X * outputDimensions + index.Y % outputDimensions) * shape.Area + index.Z;

            outGradient[outGradientIndex] = inGradient[inGradientIndex];
        }

        private static void ForwardKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, Shape shape, int outputDimensions)
        {
            int inputIndex = (index.X * shape.Dimensions + index.Y) * shape.Area + index.Z;
            int outputIndex = (index.X * outputDimensions + index.Y % outputDimensions) * shape.Area + index.Z;

            Atomic.Add(ref output[outputIndex], input[inputIndex]);
        }
    }
}
