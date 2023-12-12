using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.Layers.SkipConnection
{
    /// <summary>
    /// The <see cref="Concatenate"/> class is a <see cref="Layer"/> for combining a set of <see cref="FeatureMap"/>s from the previous
    /// <see cref="Layer"/> with the <see cref="FeatureMap"/>s from its corresponding <see cref="Fork"/>.
    /// </summary>
    public class Out : Layer, IEndpoint
    {
        private Vector _skipConnection;
        private TensorShape _skipShape;

        [JsonProperty] public int ID { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Concatenate"/> class.
        /// </summary>
        public Out() : base(1, 1)
        {
        }

        [OnDeserialized]
        private void OnDeserialized(StreamingContext context)
        {
            if(Fork.Splits.TryGetValue(ID, out var split))
            {
                split.Connect(this);
            }
            else
            {
                throw new Exception("Split source cannot be found.");
            }
        }

        /// <inheritdoc/>
        public override string Name => "Skip Out Layer";

        /// <inheritdoc/>
        public override void Backwards(int batchSize, bool update)
        {
            Index1D index = new(batchSize * _skipShape.Volume);
            GPUManager.CopyAction(index, _buffers.InGradient, _skipConnection.GetArrayViewEmpty());

            Synchronize();
            _skipConnection.Release();
        }

        /// <summary>
        /// Connects the <see cref="Concatenate"/> with its <see cref="Fork"/> sharing the <see cref="FeatureMap"/>s
        /// between them.
        /// </summary>
        /// <param name="inputs">The split outputs of the <see cref="Fork"/>.</param>
        /// <param name="outGradients">The split inGradients of the <see cref="Fork"/>.</param>
        public void Connect(Vector skipConnection, TensorShape skipInputShape, int id)
        {
            _skipConnection = skipConnection;
            _skipShape = skipInputShape;
            ID = id;
        }

        /// <inheritdoc/>
        public override void Forward(int batchSize)
        {
            Index1D index = new(batchSize * _skipShape.Volume);
            GPUManager.CopyAction(index, _skipConnection.GetArrayViewEmpty(), _buffers.Output);

            Synchronize();
            _skipConnection.Release();
        }

        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShape, PairedBuffers buffers, int batchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            _outputShape = _skipShape;

            _buffers = buffers;

            buffers.OutputDimensionArea(_outputShape.Volume);
            return _outputShape;
        }
    }
}