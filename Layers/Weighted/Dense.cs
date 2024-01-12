using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Weighted
{
    /// <summary>
    /// The <see cref="Dense"/> layer is a <see cref="WeightedLayer"/> that applies a filter the output of the previous <see cref="Layer"/> creating an
    /// output whose dimension and length are both 1.
    /// </summary>
    public class Dense : WeightedLayer
    {
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> s_backwardsFilterAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(DenseFilterKernel);

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> s_backwardsOutAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(DenseGradientKernel);

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> s_forwardAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(DenseKernel);

        [JsonProperty] private int _outputUnits;

        /// <summary>
        /// Initializes a new instance of the <see cref="Dense"/> class.
        /// </summary>
        /// <param name="outputUnits">The number of units to output when performing a forward pass with <see cref="Dense"/>.</param>
        /// <param name="weight">The initial <see cref="Dense"/> layer <see cref="Weights"/>.</param>
        /// <param name="bias">The initial bias <see cref="Weights"/>. Null if bias should not be applied to the layer.</param>
        public Dense(int outputUnits, Weights weight, Weights bias) : base(0, 0, weight, bias)
        {
            _outputUnits = outputUnits;
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor] private Dense() { }

        /// <inheritdoc/>
        public override string Name => "Dense Layer";

        private LayerInfo Info => (LayerInfo)LayerInfo;

        protected override int WeightLength => InputShape.Dimensions * InputShape.Area * _outputUnits;

        /// <inheritdoc/>
        protected override void ForwardChild(int batchSize)
        {
            Index1D copyIndex = new(InputShape.Volume * batchSize);
            GPUManager.CopyAction(copyIndex, Views.Input, _inputCopy.GetArrayViewEmpty());

            Views.Output.SubView(0, _outputUnits * batchSize).MemSetToZero();

            Index3D index = new(InputShape.Volume, batchSize, _outputUnits);
            s_forwardAction(index, Views.Input, Views.Output, _weights.WeightsView(), Info);
        }

        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int maxBatchSize)
        {
            if (Initialized)
                return OutputShape;
            Initialized = true;

            BaseStartup(inputShape, views);

            _inputCopy = new Vector(InputShape.Volume * maxBatchSize);

            return OutputShape;
        }

        private static void DenseFilterKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> input, ArrayView<float> filterGradient, LayerInfo info)
        {
            int inputArea = info.ExpansionArea * info.ExpansionDimensions;

            Atomic.Add(ref filterGradient[index.X + inputArea * index.Z], inGradient[index.Z + info.ContractionArea * index.Y] * input[index.X + inputArea * index.Y]);
        }

        private static void DenseGradientKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<float> filter, LayerInfo info)
        {
            int inputArea = info.ExpansionArea * info.ExpansionDimensions;

            Atomic.Add(ref outGradient[index.X + inputArea * index.Y], inGradient[index.Z + info.ContractionArea * index.Y] * filter[index.X + inputArea * index.Z]);
        }

        private static void DenseKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, ArrayView<float> filter, LayerInfo info)
        {
            int inputArea = info.ExpansionArea * info.ExpansionDimensions;

            Atomic.Add(ref output[index.Z + info.ContractionArea * index.Y], input[index.X + inputArea * index.Y] * filter[index.X + inputArea * index.Z]);
        }

        protected override void BackwardsNoUpdate(int batchSize)
        {
            Views.OutGradient.SubView(0, batchSize * InputShape.Volume).MemSetToZero();

            Index3D index = new(InputShape.Volume, batchSize, _outputUnits);
            s_backwardsOutAction(index, Views.InGradient, Views.OutGradient, _weights.WeightsView(), Info);
        }

        protected override void BackwardsUpdate(int batchSize)
        {
            Views.OutGradient.SubView(0, batchSize * InputShape.Volume).MemSetToZero();

            Index3D index = new(InputShape.Volume, batchSize, _outputUnits);

            s_backwardsOutAction(index, Views.InGradient, Views.OutGradient, _weights.WeightsView(), Info);
            s_backwardsFilterAction(index, Views.InGradient, _inputCopy.GetArrayView(), _weights.GradientView(), Info);
        }

        private void BaseStartup(TensorShape inputShapes, PairedGPUViews views)
        {
            InputShape = inputShapes;
            OutputShape = new TensorShape(_outputUnits, 1, 1);

            LayerInfo = new LayerInfo(inputShapes, OutputShape, 1, 1);

            Views = views;
            views.OutputDimensionArea(_outputUnits);
        }
    }
}
