using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;

namespace ConvolutionalNeuralNetwork.Layers.SkipConnection
{
    /// <summary>
    /// The <see cref="Out"/> class is a <see cref="Layer"/> for taking the output from its corresponding <see cref="Fork"/>
    /// and putting it into the correct memory in the GPU.
    /// </summary>
    public class Out : Layer, IForkEndpoint
    {
        private Vector _skipConnection;
        private TensorShape _skipShape;

        /// <summary>
        /// Initializes a new instance of the <see cref="Concatenate"/> class.
        /// </summary>
        public Out() : base(1, 1)
        {
        }

        /// <inheritdoc/>
        public override string Name => "Skip Out Layer";

        /// <inheritdoc/>
        public override void Backwards(int batchSize, bool update)
        {
            Index1D index = new(batchSize * _skipShape.Volume);
            GPUManager.CopyAction(index, Views.InGradient, _skipConnection.GetArrayViewEmpty());

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
            Index1D index = new(batchSize * _skipShape.Volume);
            GPUManager.CopyAction(index, _skipConnection.GetArrayViewEmpty(), Views.Output);

            GPUManager.Accelerator.Synchronize();
            _skipConnection.Release();
        }

        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int batchSize)
        {
            if (Initialized)
                return OutputShape;
            Initialized = true;

            OutputShape = _skipShape;

            Views = views;

            views.OutputDimensionArea(OutputShape.Volume);
            return OutputShape;
        }
    }
}