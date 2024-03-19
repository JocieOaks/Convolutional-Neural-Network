using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.Layers.SkipConnection
{
    /// <summary>
    /// The <see cref="Fork"/> class is a <see cref="Layer"/> that creates two sets of the same <see cref="Tensor"/>s, sending
    /// one as input to the next <see cref="Layer"/> and sending one to an <see cref="IEndpoint"/> later in the <see cref="Network"/>.
    /// </summary>
    public class Fork : Layer
    {
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> s_backwardsAction
                    = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(BackwardsKernel);

        private readonly List<IEndpoint> _outputLayers = new();

        private readonly List<Vector> _skipConnections = new();

        private int _maxBatchSize;

        /// <summary>
        /// Initializes a new instance of the <see cref="Fork"/> class.
        /// </summary>
        public Fork() : base(1, 1)
        {
        }

        /// <inheritdoc/>
        public override string Name => "Skip Fork Layer";

        /// <inheritdoc />
        public override bool Reflexive => true;

        /// <inheritdoc/>
        public override void Backwards(int batchSize, bool update)
        {

            Index1D index = new(batchSize * OutputShape.Volume);
            foreach (var skipConnection in _skipConnections)
            {
                s_backwardsAction(index, Views.Gradient, skipConnection.GetArrayView());
            }

            GPUManager.Accelerator.Synchronize();

            foreach (var skipConnection in _skipConnections)
            {
                skipConnection.Release();
            }
        }

        /// <inheritdoc/>
        public override void Forward(int batchSize)
        {
            Index1D index = new(batchSize * OutputShape.Volume);
            foreach (var skipConnection in _skipConnections)
            {
                GPUManager.CopyAction(index, Views.Input, skipConnection.GetArrayViewEmpty());
            }

            GPUManager.Accelerator.Synchronize();

            foreach (var skipConnection in _skipConnections)
            {
                skipConnection.Release();
            }
        }

        /// <summary>
        /// Creates a new <see cref="Concatenate"/> layer that will take the copied <see cref="Tensor"/>
        /// from this <see cref="Fork"/> as one of its inputs.
        /// </summary>
        public Concatenate GetConcatenationLayer()
        {
            var concat = new Concatenate();
            _outputLayers.Add(concat);
            
            if (!Initialized) return concat;

            var skipConnection = new Vector(_maxBatchSize * InputShape.Volume);
            concat.Connect(skipConnection, InputShape);

            return concat;
        }

        /// <summary>
        /// Creates a new <see cref="Out"/> layer that will take the copied <see cref="Tensor"/>
        /// from this <see cref="Fork"/> as its input.
        /// </summary>
        public Out GetOutLayer()
        {
            var skipOut = new Out();
            _outputLayers.Add(skipOut);
            
            if (!Initialized) return skipOut;

            var skipConnection = new Vector(_maxBatchSize * InputShape.Volume);
            skipOut.Connect(skipConnection, InputShape);

            return skipOut;
        }

        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int maxBatchSize)
        {
            if (Initialized)
                return OutputShape;
            Initialized = true;

            Views = views;
            InputShape = inputShape;
            OutputShape = inputShape;
            _maxBatchSize = maxBatchSize;

            foreach (var outputLayer in _outputLayers)
            {
                var skipConnection = new Vector(maxBatchSize * inputShape.Volume);
                _skipConnections.Add(skipConnection);
                outputLayer.Connect(skipConnection, inputShape);
            }

            return inputShape;
        }

        private static void BackwardsKernel(Index1D index, ArrayView<float> gradient, ArrayView<float> skipGradient)
        {
            gradient[index.X] += skipGradient[index.X];
        }
    }
}