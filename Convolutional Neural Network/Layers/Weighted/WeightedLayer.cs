using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;


namespace ConvolutionalNeuralNetwork.Layers.Weighted
{
    /// <summary>
    /// The <see cref="WeightedLayer"/> class is a <see cref="Layer"/> that filters a <see cref="Tensor"/>
    /// using <see cref="DataTypes.Weights"/>.
    /// </summary>
    public abstract class WeightedLayer : Layer
    {
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, int, int> s_biasAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, int, int>(BiasKernel);

        private static readonly Action<Index2D, ArrayView<float>, ArrayView<float>, int, int> s_biasGradientAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, int, int>(BiasGradientKernel);

        private readonly Weights _bias;

        /// <summary>
        /// Initializes a new instance of the <see cref="WeightedLayer"/> class.
        /// </summary>
        /// <param name="filterSize">The width and height of a filter.</param>
        /// <param name="stride">The amount of movement over the image for each filter pass.</param>
        /// <param name="weights">The initial weights for the <see cref="Layer"/>'s filters.</param>
        /// <param name="bias">The initial weights for the <see cref="Layer"/>'s bias.</param>
        protected WeightedLayer(int filterSize, int stride, Weights weights, Weights bias) : base(filterSize, stride)
        {
            Weights = weights;
            _bias = bias;
        }

        /// <value> A <see cref="Vector"/> for copying the initial input <see cref="Tensor"/> to the <see cref="Layer"/> for back-propagation.</value>
        protected Vector InputCopy { get; set; }

        /// <value>The primary <see cref="DataTypes.Weights"/> used for this <see cref="Layer"/>.</value>
        protected Weights Weights { get; }
        private bool UseBias => _bias != null;

        /// <inheritdoc />
        public sealed override void Backwards(int batchSize, bool update)
        {
            if (update)
            {
                BackwardsUpdate(batchSize);

                if (UseBias)
                {
                    Index2D biasIndex = new(OutputShape.Dimensions, batchSize);
                    s_biasGradientAction(biasIndex, _bias.GradientView(), Views.InGradient, OutputShape.Dimensions, OutputShape.Area);
                }

                GPUManager.Accelerator.Synchronize();
                BackwardsUpdateFinish();
                if (UseBias)
                {
                    _bias.ReleaseGradient();
                }
            }
            else
            {
                BackwardsNoUpdate(batchSize);
                GPUManager.Accelerator.Synchronize();
                BackwardsNoUpdateFinish();
            }
        }

        /// <inheritdoc />
        public sealed override void Forward(int batchSize)
        {
            ForwardChild(batchSize);
            if (UseBias)
            {
                Index3D biasIndex = new(OutputShape.Area, OutputShape.Dimensions, batchSize);
                s_biasAction(biasIndex, Views.Output, _bias.WeightsView(), OutputShape.Dimensions, OutputShape.Area);
            }

            GPUManager.Accelerator.Synchronize();

            ForwardFinish();

            if (UseBias)
            {
                _bias.ReleaseWeights();
            }
        }

        /// <summary>
        /// Perform back-propagation through the <see cref="Layer"/> without updating <see cref="Weights"/>.
        /// </summary>
        protected abstract void BackwardsNoUpdate(int batchSize);

        /// <summary>
        /// Called after back-propagation has completed.
        /// </summary>
        protected virtual void BackwardsNoUpdateFinish()
        {
            Weights.ReleaseWeights();
        }

        /// <summary>
        /// Perform back-propagation through the <see cref="Layer"/>, updating <see cref="Weights"/>.
        /// </summary>
        protected abstract void BackwardsUpdate(int batchSize);

        /// <summary>
        /// Called after back-propagation has completed.
        /// </summary>
        protected virtual void BackwardsUpdateFinish()
        {
            InputCopy.Release();
            Weights.ReleaseGradient();
            Weights.ReleaseWeights();
        }

        /// <summary>
        /// Forward propagate through the <see cref="Layer"/> using <see cref="Weights"/> to filter the input <see cref="Tensor"/>.
        /// Called before bias is applied.
        /// </summary>
        /// <param name="batchSize"></param>
        protected abstract void ForwardChild(int batchSize);

        /// <summary>
        /// Called after forward propagation has completed.
        /// </summary>
        protected virtual void ForwardFinish()
        {
            InputCopy.Release();
            Weights.ReleaseWeights();
        }

        private static void BiasGradientKernel(Index2D index, ArrayView<float> biasGradient, ArrayView<float> inGradient, int dimensions, int length)
        {
            float sum = 0;
            for (int i = 0; i < length; i++)
            {
                sum += inGradient[(index.Y * dimensions + index.X) * length + i];
            }
            Atomic.Add(ref biasGradient[index.X], sum);
        }

        private static void BiasKernel(Index3D index, ArrayView<float> value, ArrayView<float> bias, int dimensions, int length)
        {
            Atomic.Add(ref value[(index.Z * dimensions + index.Y) * length + index.X], bias[index.Y]);
        }
    }
}
