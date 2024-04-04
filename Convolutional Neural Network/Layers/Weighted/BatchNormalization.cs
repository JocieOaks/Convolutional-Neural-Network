using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Weighted
{
    /// <summary>
    /// The <see cref="BatchNormalization"/> class is a <see cref="Layer"/> for normalizing batches of <see cref="Tensor"/>s
    /// so that their mean is 0 and standard deviation 1.
    /// </summary>
    public class BatchNormalization : WeightedLayer
    {
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, NormalizationViews, TensorShape> s_backwardsAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, NormalizationViews, TensorShape>(WeightsAndGradientKernel);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, NormalizationViews, TensorShape> s_gradientAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, NormalizationViews, TensorShape>(GradientsKernel);
        private static readonly Action<Index3D, ArrayView<float>, NormalizationViews, TensorShape> s_normalizeAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, NormalizationViews, TensorShape>(NormalizeKernel);
        private static readonly Action<Index3D, ArrayView<float>, NormalizationViews, TensorShape> s_sumAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, NormalizationViews, TensorShape>(SumKernel);
        private static readonly Action<Index3D, ArrayView<float>, NormalizationViews, TensorShape> s_varianceAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, NormalizationViews, TensorShape>(VarianceKernel);
        private static readonly Action<Index1D, NormalizationViews, float> s_meanAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, NormalizationViews, float>(MeanKernel);
        private static readonly Action<Index1D, NormalizationViews, float> s_sigmaAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, NormalizationViews, float>(SigmaKernel);
        private static readonly Action<Index1D, NormalizationViews, float> s_meanSigmaGradientAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, NormalizationViews, float>(MeanSigmaGradientKernel);

        private readonly Weights _bias;
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

        /// <inheritdoc />
        [JsonIgnore] public override bool Reflexive => true;

        /// <inheritdoc/>
        protected override void BackwardsUpdate(int batchSize)
        {
            NormalizationViews normalizationViews = new()
            {
                Mean = _mean.GetArrayView(),
                Sigma = _sigma.GetArrayView(),
                Weight = Weights.WeightsView(),
                Bias = _bias.WeightsView(),
                MeanGradient = _meanGradient.GetArrayView(),
                SigmaGradient = _sigmaGradient.GetArrayView(),
                WeightGradient = Weights.GradientView(),
                BiasGradient = _bias.GradientView()
            };

            ArrayView<float> input = InputCopy.GetArrayView();

            Index3D index = new(InputShape.Area, InputShape.Dimensions, batchSize);
            s_gradientAction(index, input, Views.Gradient, normalizationViews, InputShape);

            GPUManager.Accelerator.Synchronize();


            Index1D dimensionIndex = new(InputShape.Dimensions);
            s_meanSigmaGradientAction(dimensionIndex, normalizationViews, 1f / (batchSize * InputShape.Area));

            GPUManager.Accelerator.Synchronize();


            s_backwardsAction(index, input, Views.Gradient, normalizationViews, InputShape);
        }

        /// <inheritdoc />
        protected override void BackwardsUpdateFinish()
        {
            _mean.Release();
            _sigma.Release();
            Weights.ReleaseWeights();
            _bias.ReleaseWeights();
            _meanGradient.Release();
            _sigmaGradient.Release();
            Weights.ReleaseGradient();
            _bias.ReleaseGradient();
            InputCopy.Release();
        }

        /// <inheritdoc />
        protected override void BackwardsNoUpdateFinish()
        {
            BackwardsUpdateFinish();
        }

        /// <inheritdoc/>
        protected override void ForwardChild(int batchSize)
        {
            Index1D copyIndex = new(batchSize * InputShape.Volume);
            GPUManager.CopyAction(copyIndex, Views.Input, InputCopy.GetArrayViewEmpty());


            NormalizationViews normalizationViews = new()
            {
                Mean = _mean.GetArrayViewZeroed(),
                Sigma = _sigma.GetArrayViewZeroed(),
                Weight = Weights.WeightsView(),
                Bias = _bias.WeightsView()
            };

            Index3D index = new(InputShape.Area, InputShape.Dimensions, batchSize);

            s_sumAction(index, Views.Input, normalizationViews, InputShape);

            GPUManager.Accelerator.Synchronize();


            Index1D dimensionIndex = new(InputShape.Dimensions);
            float inverseArea = 1f / (batchSize * InputShape.Area);
            s_meanAction(dimensionIndex, normalizationViews, inverseArea);

            GPUManager.Accelerator.Synchronize();


            s_varianceAction(index, Views.Input, normalizationViews, InputShape);

            GPUManager.Accelerator.Synchronize();


            s_sigmaAction(dimensionIndex, normalizationViews, inverseArea);

            GPUManager.Accelerator.Synchronize();


            s_normalizeAction(index, Views.Input, normalizationViews, InputShape);
        }

        /// <inheritdoc />
        protected override void ForwardFinish()
        {
            InputCopy.Release();
            _mean.Release();
            _sigma.Release();
            Weights.ReleaseWeights();
            _bias.ReleaseWeights();
        }

        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int maxBatchSize)
        {
            if (Initialized)
                return OutputShape;
            Initialized = true;

            BaseStartup(inputShape, views);

            _mean = new Vector(InputShape.Dimensions);
            _meanGradient = new Vector(InputShape.Dimensions);
            _sigma = new Vector(InputShape.Dimensions);
            _sigmaGradient = new Vector(InputShape.Dimensions);
            InputCopy = new Vector(maxBatchSize * inputShape.Volume);

            return OutputShape;
        }

        private static void WeightsAndGradientKernel(Index3D index, ArrayView<float> input, ArrayView<float> gradient, NormalizationViews values, TensorShape shape)
        {
            int ind = index.Z * shape.Volume + index.Y * shape.Area + index.X;
            gradient[ind] = XMath.Clamp(gradient[ind] * values.Weight[index.Y] / values.Sigma[index.Y] + values.SigmaGradient[index.Y] * (input[ind] - values.Mean[index.Y]) + values.MeanGradient[index.Y], -1, 1);
        }

        private static void MeanSigmaGradientKernel(Index1D index, NormalizationViews values, float inverseArea)
        {
            values.MeanGradient[index] = -inverseArea * values.BiasGradient[index] * values.Weight[index] / values.Sigma[index];
            values.SigmaGradient[index] *= inverseArea * XMath.Pow(values.Sigma[index], -3) * values.Weight[index];
        }

        private static void MeanKernel(Index1D index, NormalizationViews values, float inverseArea)
        {
            values.Mean[index] = values.Mean[index] * inverseArea;
        }

        private static void SigmaKernel(Index1D index, NormalizationViews values, float inverseArea)
        {
            values.Sigma[index] = XMath.Pow(values.Sigma[index] * inverseArea + Utility.ASYMPTOTE_ERROR_CORRECTION, 0.5f);
        }

        private static void NormalizeKernel(Index3D index, ArrayView<float> input, NormalizationViews values, TensorShape shape)
        {
            int ind = index.Z * shape.Volume + index.Y * shape.Area + index.X;
            input[ind] = (input[ind] - values.Mean[index.Y]) * values.Weight[index.Y] / values.Sigma[index.Y] + values.Bias[index.Y];
        }

        private static void GradientsKernel(Index3D index, ArrayView<float> input, ArrayView<float> inGradient, NormalizationViews values, TensorShape shape)
        {
            int ind = index.Z * shape.Volume + index.Y * shape.Area + index.X;

            float meanOffset = input[ind] - values.Mean[index.Y];
            float gradient = inGradient[ind];

            float normalized = meanOffset * values.Weight[index.Y] / values.Sigma[index.Y] + values.Bias[index.Y];

            Atomic.Add(ref values.SigmaGradient[index.Y], gradient * meanOffset);
            Atomic.Add(ref values.WeightGradient[index.Y], gradient * normalized);
            Atomic.Add(ref values.BiasGradient[index.Y], gradient);
        }

        private static void SumKernel(Index3D index, ArrayView<float> input, NormalizationViews values, TensorShape shape)
        {
            Atomic.Add(ref values.Mean[index.Y], input[index.Z * shape.Volume + index.Y * shape.Area + index.X]);
        }

        private static void VarianceKernel(Index3D index, ArrayView<float> input, NormalizationViews values, TensorShape shape)
        {
            float difference = input[index.Z * shape.Volume + index.Y * shape.Area + index.X] - values.Mean[index.Y];
            Atomic.Add(ref values.Sigma[index.Y], difference * difference);
        }

        /// <inheritdoc />
        protected override void BackwardsNoUpdate(int batchSize)
        {
            BackwardsUpdate(batchSize);
        }

        private readonly struct NormalizationViews
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