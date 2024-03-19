using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.Layers.SkipConnection
{
    /// <summary>
    /// The <see cref="Concatenate"/> class is a <see cref="Layer"/> for concatenating a set of <see cref="Tensor"/>s from the previous
    /// <see cref="Layer"/> with the <see cref="Tensor"/>s from its corresponding <see cref="Fork"/>.
    /// </summary>
    public class Concatenate : Layer, IEndpoint
    {
        private Vector _skipConnection;
        private TensorShape _skipShape;

        /// <summary>
        /// Initializes a new instance of the <see cref="Concatenate"/> class.
        /// </summary>
        public Concatenate() : base(1, 1)
        {
        }

        /// <inheritdoc/>
        public override string Name => "Skip Concatenation Layer";

        /// <inheritdoc/>
        public override void Backwards(int batchSize, bool update)
        {
            Index3D index = new(InputShape.Area, InputShape.Dimensions, batchSize);
            s_backwardsAction(index, Views.InGradient, Views.OutGradient, OutputShape, InputShape, 0);

            index = new(_skipShape.Area, _skipShape.Dimensions, batchSize);
            s_backwardsAction(index, Views.InGradient, _skipConnection.GetArrayView(), OutputShape, _skipShape, InputShape.Dimensions);

            GPUManager.Accelerator.Synchronize();
            _skipConnection.Release();
        }

        /// <inheritdoc />
        public void Connect(Vector skipConnection, TensorShape skipInputShape)
        {
            _skipConnection = skipConnection;
            _skipShape = skipInputShape;
        }

        /// <inheritdoc/>
        public override void Forward(int batchSize)
        {
            Index3D index = new(InputShape.Area, InputShape.Dimensions, batchSize);
            s_forwardAction(index, Views.Input, Views.Output, InputShape, OutputShape, 0);
            index = new(_skipShape.Area, _skipShape.Dimensions, batchSize);
            s_forwardAction(index, _skipConnection.GetArrayView(), Views.Output, _skipShape, OutputShape, InputShape.Dimensions);

            GPUManager.Accelerator.Synchronize();
            _skipConnection.Release();
        }

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, TensorShape, TensorShape, int> s_forwardAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, TensorShape, TensorShape, int>(ConcatenationKernel);

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, TensorShape, TensorShape, int> s_backwardsAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, TensorShape, TensorShape, int>(ConcatenationGradientKernel);

        private static void ConcatenationKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, TensorShape inputShape, TensorShape outputShape, int offset)
        {
            output[(index.Z * outputShape.Dimensions + index.Y + offset) * outputShape.Area + index.X] = input[(index.Z * inputShape.Dimensions + index.Y) * inputShape.Area + index.X];
        }

        private static void ConcatenationGradientKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, TensorShape inShape, TensorShape outShape, int offset)
        {
            outGradient[(index.Z * outShape.Dimensions + index.Y) * outShape.Area + index.X] = inGradient[(index.Z * inShape.Dimensions + index.Y + offset) * inShape.Area + index.X];
        }

        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int batchSize)
        {
            if (Initialized)
                return OutputShape;
            Initialized = true;

            if (inputShape.Area != _skipShape.Area)
            {
                throw new ArgumentException("Input shapes do not match.");
            }

            OutputShape = new TensorShape(inputShape.Width, inputShape.Length, inputShape.Dimensions + _skipShape.Dimensions);

            Views = views;
            InputShape = inputShape;

            views.OutputDimensionArea(OutputShape.Volume);

            return OutputShape;
        }
    }
}