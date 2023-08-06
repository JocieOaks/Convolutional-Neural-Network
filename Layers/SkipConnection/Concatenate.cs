using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.Layers.SkipConnection
{
    /// <summary>
    /// The <see cref="Concatenate"/> class is a <see cref="Layer"/> for combining a set of <see cref="FeatureMap"/>s from the previous
    /// <see cref="Layer"/> with the <see cref="FeatureMap"/>s from its corresponding <see cref="Fork"/>.
    /// </summary>
    public class Concatenate : Layer, IStructuralLayer, IEndpoint
    {
        private Vector _skipConnection;
        private Shape _skipShape;

        [JsonProperty] public int ID { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Concatenate"/> class.
        /// </summary>
        public Concatenate() : base(1, 1)
        {
        }

        [OnDeserialized]
        private void OnDeserialized(StreamingContext context)
        {
            if (Fork.Splits.TryGetValue(ID, out var split))
            {
                split.Connect(this);
            }
            else
            {
                throw new Exception("Split source cannot be found.");
            }
        }

        /// <inheritdoc/>
        public override string Name => "Skip Concatenation Layer";

        /// <inheritdoc/>
        public override void Backwards(int batchSize, bool update)
        {
            Index3D index = new(_inputShape.Area, _inputShape.Dimensions, batchSize);
            s_backwardsAction(index, _buffers.InGradient, _buffers.OutGradient, _outputShape, _inputShape, 0);

            index = new(_skipShape.Area, _skipShape.Dimensions, batchSize);
            s_backwardsAction(index, _buffers.InGradient, _skipConnection.GetArrayView<float>(), _outputShape, _skipShape, _inputShape.Dimensions);

            Synchronize();
            _skipConnection.DecrementLiveCount();
        }

        /// <summary>
        /// Connects the <see cref="Concatenate"/> with its <see cref="Fork"/> sharing the <see cref="FeatureMap"/>s
        /// between them.
        /// </summary>
        /// <param name="inputs">The split outputs of the <see cref="Fork"/>.</param>
        /// <param name="outGradients">The split inGradients of the <see cref="Fork"/>.</param>
        public void Connect(Vector skipConnection, Shape skipInputShape, int id)
        {
            _skipConnection = skipConnection;
            _skipShape = skipInputShape;
            ID = id;
        }

        /// <inheritdoc/>
        public override void Forward(int batchSize)
        {
            Index3D index = new(_inputShape.Area, _inputShape.Dimensions, batchSize);
            s_forwardAction(index, _buffers.Input, _buffers.Output, _inputShape, _outputShape, 0);
            index = new(_skipShape.Area, _skipShape.Dimensions, batchSize);
            s_forwardAction(index, _skipConnection.GetArrayView<float>(), _buffers.Output, _skipShape, _outputShape, _inputShape.Dimensions);

            Synchronize();
            _skipConnection.DecrementLiveCount();
        }

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, Shape, Shape, int> s_forwardAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, Shape, Shape, int>(ConcatenationKernel);

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, Shape, Shape, int> s_backwardsAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, Shape, Shape, int>(ConcatenationGradientKernel);

        private static void ConcatenationKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, Shape inputShape, Shape outputShape, int offset)
        {
            output[(index.Z * outputShape.Dimensions + index.Y + offset) * outputShape.Area + index.X] = input[(index.Z * inputShape.Dimensions + index.Y) * inputShape.Area + index.X];
        }

        private static void ConcatenationGradientKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, Shape inShape, Shape outShape, int offset)
        {
            outGradient[(index.Z * outShape.Dimensions + index.Y) * outShape.Area + index.X] = inGradient[(index.Z * inShape.Dimensions + index.Y + offset) * inShape.Area + index.X];
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override Shape Startup(Shape inputShape, IOBuffers buffers, int batchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            if (inputShape.Area != _skipShape.Area)
            {
                throw new ArgumentException("Input shapes do not match.");
            }

            _outputShape = new Shape(inputShape.Width, inputShape.Length, inputShape.Dimensions + _skipShape.Dimensions);

            _buffers = buffers;
            _inputShape = inputShape;

            buffers.OutputDimensionArea(_outputShape.Volume);

            return _outputShape;
        }
    }
}