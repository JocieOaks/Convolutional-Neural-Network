using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Weighted
{
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

        /// <summary>
        /// Gets the <see cref="LayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="LayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="LayerInfo"/> corresponding to an input dimension.</returns>
        private LayerInfo Info => (LayerInfo)_layerInfo;

        protected override int WeightLength => _inputShape.Dimensions * _inputShape.Area * _outputUnits;

        /// <inheritdoc/>
        protected override void ForwardChild(int batchSize)
        {
            Index1D copyIndex = new(_inputShape.Volume * batchSize);
            GPUManager.CopyAction(copyIndex, _buffers.Input, _inputCopy.GetArrayViewEmpty<float>());

            _buffers.Output.SubView(0, _outputUnits * batchSize).MemSetToZero();

            Index3D index = new(_inputShape.Volume, batchSize, _outputUnits);
            s_forwardAction(index, _buffers.Input, _buffers.Output, _weights.WeightsGPU<float>(), Info);
        }

        /// <inheritdoc/>
        public override Shape Startup(Shape inputShape, IOBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            BaseStartup(inputShape, buffers);

            _inputCopy = new Vector(_inputShape.Volume * maxBatchSize);

            return _outputShape;
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
            _buffers.OutGradient.SubView(0, batchSize * _inputShape.Volume).MemSetToZero();

            Index3D index = new(_inputShape.Volume, batchSize, _outputUnits);
            s_backwardsOutAction(index, _buffers.InGradient, _buffers.OutGradient, _weights.WeightsGPU<float>(), Info);
        }

        protected override void BackwardsUpdate(int batchSize)
        {
            _buffers.OutGradient.SubView(0, batchSize * _inputShape.Volume).MemSetToZero();

            Index3D index = new(_inputShape.Volume, batchSize, _outputUnits);

            s_backwardsOutAction(index, _buffers.InGradient, _buffers.OutGradient, _weights.WeightsGPU<float>(), Info);
            s_backwardsFilterAction(index, _buffers.InGradient, _inputCopy.GetArrayView<float>(), _weights.GradientGPU<float>(), Info);
        }

        private void BaseStartup(Shape inputShapes, IOBuffers buffers)
        {
            _inputShape = inputShapes;
            _outputShape = new Shape(_outputUnits, 1, 1);

            _layerInfo = new LayerInfo(inputShapes, _outputShape, 1, 1);

            _buffers = buffers;
            buffers.OutputDimensionArea(_outputUnits);
        }
    }
}
