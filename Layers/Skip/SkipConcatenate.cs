using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.Layers.Skip
{
    /// <summary>
    /// The <see cref="SkipConcatenate"/> class is a <see cref="Layer"/> for combining a set of <see cref="FeatureMap"/>s from the previous
    /// <see cref="Layer"/> with the <see cref="FeatureMap"/>s from its corresponding <see cref="SkipSplit"/>.
    /// </summary>
    public class SkipConcatenate : Layer, IStructuralLayer, ISkipEndpoint
    {
        private Vector _skipConnection;
        private Shape _skipShape;

        [JsonProperty] public int ID { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="SkipConcatenate"/> class.
        /// </summary>
        public SkipConcatenate() : base(1, 1)
        {
        }

        [OnDeserialized]
        private void OnDeserialized(StreamingContext context)
        {
            if (SkipSplit.Splits.TryGetValue(ID, out var split))
            {
                split.Connect(this);
            }
            else
            {
                throw new Exception("Split source cannot be found.");
            }
        }

        /// <inheritdoc/>
        public override string Name => "Concatenation Layer";

        /// <inheritdoc/>
        public override void Backwards(int batchSize)
        {
            Index3D index = new(batchSize, _outputShape.Dimensions, _outputShape.Area);
            s_backwardsAction(index, _buffers.InGradient, _buffers.OutGradient, _skipConnection.GetArrayView<float>(), _inputShape, _skipShape.Dimensions);

            Synchronize();
            _skipConnection.DecrementLiveCount();
        }

        /// <summary>
        /// Connects the <see cref="SkipConcatenate"/> with its <see cref="SkipSplit"/> sharing the <see cref="FeatureMap"/>s
        /// between them.
        /// </summary>
        /// <param name="inputs">The split outputs of the <see cref="SkipSplit"/>.</param>
        /// <param name="outGradients">The split inGradients of the <see cref="SkipSplit"/>.</param>
        public void Connect(Vector skipConnection, Shape skipInputShape, int id)
        {
            _skipConnection = skipConnection;
            _skipShape = skipInputShape;
            ID = id;
        }

        /// <inheritdoc/>
        public override void Forward(int batchSize)
        {
            Index3D index = new(batchSize, _outputShape.Dimensions, _outputShape.Area);
            s_forwardAction(index, _buffers.Input, _skipConnection.GetArrayView<float>(), _buffers.Output, _inputShape, _skipShape.Dimensions);

            Synchronize();
            _skipConnection.DecrementLiveCount();
        }

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, Shape, int> s_forwardAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, Shape, int>(ForwardKernel);

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, Shape, int> s_backwardsAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, Shape, int>(BackwardsKernel);

        private static void ForwardKernel(Index3D index, ArrayView<float> input, ArrayView<float> skip, ArrayView<float> output, Shape shape, int skipDimensions)
        {
            float value;
            if (index.Y >= shape.Dimensions)
            {
                value = skip[(index.X * skipDimensions + index.Y - shape.Dimensions) * shape.Area + index.Z];
            }
            else
            {
                value = input[(index.X * shape.Dimensions + index.Y) * shape.Area + index.Z];
            }

            output[(index.X * (shape.Dimensions + skipDimensions) + index.Y) * shape.Area + index.Z] = value;
        }

        private static void BackwardsKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<float> skipGradient, Shape shape, int skipDimensions)
        {
            float value = inGradient[(index.X * (shape.Dimensions + skipDimensions) + index.Y) * shape.Area + index.Z];

            if (index.Y >= shape.Dimensions)
            {
                skipGradient[(index.X * skipDimensions + index.Y - shape.Dimensions) * shape.Area + index.Z] = value;
            }
            else
            {
                outGradient[(index.X * shape.Dimensions + index.Y) * shape.Area + index.Z] = value;
            }
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