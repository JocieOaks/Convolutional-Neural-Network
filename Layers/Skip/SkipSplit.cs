using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.IR.Types;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.Layers.Skip
{
    /// <summary>
    /// The <see cref="SkipSplit"/> class is a <see cref="Layer"/> that creates two sets of the same <see cref="FeatureMap"/>s, sending
    /// one as input to the next <see cref="Layer"/> and sending one to a <see cref="SkipConcatenate"/> later in the <see cref="Network"/>.
    /// </summary>
    public class SkipSplit : Layer, IStructuralLayer, IUnchangedLayer
    {
        private static int s_nextID = 1;
        public static Dictionary<int, SkipSplit> Splits { get; } = new Dictionary<int, SkipSplit>();

        /// <inheritdoc/>
        public override string Name => "Skip Connection Layer";

        private List<ISkipEndpoint> _outputLayers = new();
        private List<Vector> _skipConnections = new();
        private int _maxBatchSize;
        [JsonProperty] public int ID { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="SkipSplit"/> class.
        /// </summary>
        public SkipSplit() : base(1, 1)
        {
            ID = s_nextID++;
            Splits[ID] = this;
        }

        [JsonConstructor] private SkipSplit(int id)
        {
            ID = id;
            Splits[id] = this;
        }

        /// <summary>
        /// Gives the corresponding <see cref="SkipConcatenate"/> layer that connects to this <see cref="SkipSplit"/>, creating
        /// it if it does not already exist.
        /// </summary>
        /// <returns>Returns the <see cref="SkipConcatenate"/>.</returns>
        public SkipConcatenate GetConcatenationLayer()
        {
            var concat = new SkipConcatenate();
            _outputLayers.Add(concat);
            if (_ready)
            {
                var skipConnection = new Vector(_maxBatchSize * _inputShape.Volume);
                concat.Connect(skipConnection, _inputShape, ID);
            }

            return concat;
        }

        public void Connect(ISkipEndpoint endpoint)
        {
            _outputLayers.Add(endpoint);
            if (_ready)
            {
                var skipConnection = new Vector(_maxBatchSize * _inputShape.Volume);
                endpoint.Connect(skipConnection, _inputShape, ID);
            }
        }

        public SkipOut GetOutLayer()
        {
            var skipOut = new SkipOut();
            _outputLayers.Add(skipOut);
            if (_ready)
            {
                var skipConnection = new Vector(_maxBatchSize * _inputShape.Volume);
                skipOut.Connect(skipConnection, _inputShape, ID);
            }

            return skipOut;
        }


        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> s_backwardsAction
            = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(BackwardsKernel);

        /// <inheritdoc/>
        public override void Backwards(int batchSize, bool update)
        {

            Index1D index = new(batchSize * _outputShape.Volume);
            foreach (var skipConnection in _skipConnections)
            {
                s_backwardsAction(index, _buffers.Gradient, skipConnection.GetArrayView<float>());
            }

            Synchronize();

            foreach (var skipConnection in _skipConnections)
            {
                skipConnection.DecrementLiveCount();
            }
        }

        /// <inheritdoc/>
        public override void Forward(int batchSize)
        {
            Index1D index = new(batchSize * _outputShape.Volume);
            foreach (var skipConnection in _skipConnections)
            {
                GPUManager.CopyAction(index, _buffers.Input, skipConnection.GetArrayViewEmpty<float>());
            }

            Synchronize();

            foreach (var skipConnection in _skipConnections)
            {
                skipConnection.DecrementLiveCount();
            }
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override Shape Startup(Shape inputShape, IOBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            _buffers = buffers;
            _inputShape = inputShape;
            _outputShape = inputShape;
            _maxBatchSize = maxBatchSize;

            foreach (var outputLayer in _outputLayers)
            {
                var skipConnection = new Vector(maxBatchSize * inputShape.Volume);
                _skipConnections.Add(skipConnection);
                outputLayer.Connect(skipConnection, inputShape, ID);
            }

            return inputShape;
        }

        private static void BackwardsKernel(Index1D index, ArrayView<float> gradient, ArrayView<float> skipGradient)
        {
            gradient[index.X] += skipGradient[index.X];
        }
    }
}