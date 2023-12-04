using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Weighted
{
    /// <summary>
    /// The <see cref="BatchNormalization"/> class is a <see cref="Layer"/> for normalizing batches of <see cref="FeatureMap"/>s
    /// so that their mean is 0 and standard deviation 1.
    /// </summary>
    public class BatchNormalization : WeightedLayer, IReflexiveLayer
    {
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, Views, TensorShape> s_backwardsAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, Views, TensorShape>(WeightsAndGradientKernel);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, Views, TensorShape> s_gradientAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, Views, TensorShape>(GradientsKernel);
        private static readonly Action<Index3D, ArrayView<float>, Views, TensorShape> s_normalizeAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, Views, TensorShape>(NormalizeKernel);
        private static readonly Action<Index3D, ArrayView<float>, Views, TensorShape> s_sumAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, Views, TensorShape>(SumKernel);
        private static readonly Action<Index3D, ArrayView<float>, Views, TensorShape> s_varianceAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, Views, TensorShape>(VarianceKernel);
        private static readonly Action<Index1D, Views, float> s_meanAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, Views, float>(MeanKernel);
        private static readonly Action<Index1D, Views, float> s_sigmaAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, Views, float>(SigmaKernel);
        private static readonly Action<Index1D, Views, float> s_meanSigmaGradientAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, Views, float>(MeanSigmaGradientKernel);

        private Weights _bias;
        private Vector _mean;
        private Vector _meanGradient;
        private Vector _sigma;
        private Vector _sigmaGradient;

        /// <summary>
        /// Initializes a new instance of the <see cref="BatchNormalization"/> class.
        /// </summary>
        public BatchNormalization(Weights weights, Weights bias) : base(1, 1, weights, null)
        {
            _bias = bias;
        }

        /// <inheritdoc/>
        [JsonIgnore] public override string Name => "Batch Normalization Layer";

        protected override int WeightLength => _outputShape.Dimensions;

        /// <inheritdoc/>
        protected override void BackwardsUpdate(int batchSize)
        {
            Views views = new()
            {
                Mean = _mean.GetArrayView(),
                Sigma = _sigma.GetArrayView(),
                Weight = _weights.WeightsGPU(),
                Bias = _bias.WeightsGPU(),
                MeanGradient = _meanGradient.GetArrayView(),
                SigmaGradient = _sigmaGradient.GetArrayView(),
                WeightGradient = _weights.GradientGPU(),
                BiasGradient = _bias.GradientGPU()
            };

            Index3D index = new(_inputShape.Area, _inputShape.Dimensions, batchSize);
            s_gradientAction(index, _inputCopy.GetArrayView(), _buffers.Gradient, views, _inputShape);

            Synchronize();


            Index1D dimensionIndex = new(_inputShape.Dimensions);
            s_meanSigmaGradientAction(dimensionIndex, views, 1f / (batchSize * _inputShape.Area));

            Synchronize();


            s_backwardsAction(index, _inputCopy.GetArrayView(), _buffers.Gradient, views, _inputShape);
        }

        protected override void BackwardsUpdateFinish()
        {
            _mean.DecrementLiveCount();
            _sigma.DecrementLiveCount();
            _weights.DecrementLiveWeights();
            _bias.DecrementLiveWeights();
            _meanGradient.DecrementLiveCount();
            _sigmaGradient.DecrementLiveCount();
            _weights.DecrementLiveGradient();
            _bias.DecrementLiveGradient();
            _inputCopy.DecrementLiveCount(2);
        }

        protected override void BackwardsNoUpdateFinish()
        {
            BackwardsUpdateFinish();
        }

        /// <inheritdoc/>
        protected override void ForwardChild(int batchSize)
        {
            Index1D copyIndex = new(batchSize * _inputShape.Volume);
            GPUManager.CopyAction(copyIndex, _buffers.Input, _inputCopy.GetArrayViewEmpty());


            Views views = new()
            {
                Mean = _mean.GetArrayViewZeroed(),
                Sigma = _sigma.GetArrayViewZeroed(),
                Weight = _weights.WeightsGPU(),
                Bias = _bias.WeightsGPU()
            };

            Index3D index = new(_inputShape.Area, _inputShape.Dimensions, batchSize);

            s_sumAction(index, _buffers.Input, views, _inputShape);

            Synchronize();


            Index1D dimensionIndex = new(_inputShape.Dimensions);
            float inverseArea = 1f / (batchSize * _inputShape.Area);
            s_meanAction(dimensionIndex, views, inverseArea);

            Synchronize();


            s_varianceAction(index, _buffers.Input, views, _inputShape);

            Synchronize();


            s_sigmaAction(dimensionIndex, views, inverseArea);

            Synchronize();


            s_normalizeAction(index, _buffers.Input, views, _inputShape);
        }

        protected override void ForwardFinish()
        {
            _inputCopy.DecrementLiveCount();
            _mean.DecrementLiveCount();
            _sigma.DecrementLiveCount();
            _weights.DecrementLiveWeights();
            _bias.DecrementLiveWeights();
        }

        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShape, PairedBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            BaseStartup(inputShape, buffers);

            _mean = new Vector(_inputShape.Dimensions);
            _meanGradient = new Vector(_inputShape.Dimensions);
            _sigma = new Vector(_inputShape.Dimensions);
            _sigmaGradient = new Vector(_inputShape.Dimensions);
            _inputCopy = new Vector(maxBatchSize * inputShape.Volume);

            return _outputShape;
        }

        /// <summary>
        /// An ILGPU kernel to calculate the gradients for backpropagating the previous layer.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="gradient">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the incoming
        /// gradient from the following <see cref="Layer"/>.</param>
        /// <param name="outGradient">An <see cref="ArrayView1D{T, TStride}"/> of floats to sum the outgoing gradient.
        /// Because <see cref="Color"/> cannot be summed atomically, every three floats represents a single
        /// <see cref="Color"/> in the gradient.</param>
        /// <param name="values">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s used in the equation
        /// to calculate the outGradient.</param>
        /// <param name="info">The <see cref="StaticLayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void WeightsAndGradientKernel(Index3D index, ArrayView<float> input, ArrayView<float> gradient, Views values, TensorShape shape)
        {
            int ind = index.Z * shape.Volume + index.Y * shape.Area + index.X;
            gradient[ind] = XMath.Clamp(gradient[ind] * values.Weight[index.Y] / values.Sigma[index.Y] + values.SigmaGradient[index.Y] * (input[ind] - values.Mean[index.Y]) + values.MeanGradient[index.Y], -1, 1);
        }

        private static void MeanSigmaGradientKernel(Index1D index, Views values, float inverseArea)
        {
            values.MeanGradient[index] = -inverseArea * values.BiasGradient[index] * values.Weight[index] / values.Sigma[index];
            values.SigmaGradient[index] *= inverseArea * XMath.Pow(values.Sigma[index], -3) * values.Weight[index];
        }

        private static void MeanKernel(Index1D index, Views values, float inverseArea)
        {
            values.Mean[index] = values.Mean[index] * inverseArea;
        }

        private static void SigmaKernel(Index1D index, Views values, float inverseArea)
        {
            values.Sigma[index] = XMath.Pow(values.Sigma[index] * inverseArea + Utility.ASYMPTOTEERRORCORRECTION, 0.5f);
        }

        /// <summary>
        /// An ILGPU kernel for normalizing a <see cref="FeatureMap"/>.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="normalized">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s to set for the outgoing
        /// convoluted <see cref="FeatureMap"/>.</param>
        /// <param name="values">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s used in the equation to
        /// calculate the normalized <see cref="Color"/>.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void NormalizeKernel(Index3D index, ArrayView<float> input, Views values, TensorShape shape)
        {
            int ind = index.Z * shape.Volume + index.Y * shape.Area + index.X;
            input[ind] = (input[ind] - values.Mean[index.Y]) * values.Weight[index.Y] / values.Sigma[index.Y] + values.Bias[index.Y];
        }

        private static void GradientsKernel(Index3D index, ArrayView<float> input, ArrayView<float> inGradient, Views values, TensorShape shape)
        {
            int ind = index.Z * shape.Volume + index.Y * shape.Area + index.X;

            float meanOffset = input[ind] - values.Mean[index.Y];
            float gradient = inGradient[ind];

            float normalized = meanOffset * values.Weight[index.Y] / values.Sigma[index.Y] + values.Bias[index.Y];

            Atomic.Add(ref values.SigmaGradient[index.Y], gradient * meanOffset);
            Atomic.Add(ref values.WeightGradient[index.Y], gradient * normalized);
            Atomic.Add(ref values.BiasGradient[index.Y], gradient);
        }

        private static void SumKernel(Index3D index, ArrayView<float> input, Views values, TensorShape shape)
        {
            Atomic.Add(ref values.Mean[index.Y], input[index.Z * shape.Volume + index.Y * shape.Area + index.X]);
        }

        private static void VarianceKernel(Index3D index, ArrayView<float> input, Views values, TensorShape shape)
        {
            float difference = input[index.Z * shape.Volume + index.Y * shape.Area + index.X] - values.Mean[index.Y];
            Atomic.Add(ref values.Sigma[index.Y], difference * difference);
        }

        protected override void BackwardsNoUpdate(int batchSize)
        {
            BackwardsUpdate(batchSize);
        }

        /*protected override void BiasTest(TensorShape input, TensorShape output, int batchSize)
        {
            _bias.TestFilterGradient(this, input, output, _buffers, batchSize);
        }*/

        private readonly struct Views
        {
            public ArrayView<float> Mean { get; init; }
            public ArrayView<float> Sigma { get; init; }
            public ArrayView<float> Weight { get; init; }
            public ArrayView<float> Bias { get; init; }
            public ArrayView<float> MeanGradient { get; init; }
            public ArrayView<float> SigmaGradient { get; init; }
            public ArrayView<float> WeightGradient { get; init; }
            public ArrayView<float> BiasGradient { get; init; }
        }
    }
}