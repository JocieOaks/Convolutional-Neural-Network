using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ConvolutionalNeuralNetwork.Layers.Initializers;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Collections.Generic;

namespace ConvolutionalNeuralNetwork.Layers.Weighted
{
    public class Dense : WeightedLayer, IPrimaryLayer
    {
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> s_backwardsFilterAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(BackwardsFilterKernel);

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> s_backwardsOutAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(BackwardsOutKernel);

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> s_forwardAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(ForwardKernel);

        [JsonProperty] private Weights _filters;
        private Vector _inputCopy;
        [JsonProperty] private int _outputUnits;

        /// <summary>
        /// Initializes a new instance of the <see cref="Dense"/> class.
        /// </summary>
        /// <param name="outputUnits">The number of units to output when performing a forward pass with <see cref="Dense"/>.</param>
        public Dense(int outputUnits, IWeightInitializer initializer, bool useBias = true) : base(0, 0, initializer, useBias)
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

        public void FilterTest(int inputDimensions, int batchSize, int inputSize)
        {
            (Shape input, Shape output) = FilterTestSetup(inputDimensions, batchSize, inputSize);

            _filters.TestFilterGradient(this, input, output, _buffers, batchSize);
            BiasTest(input, output, batchSize);
        }

        /// <inheritdoc/>
        protected override void ForwardChild(int batchSize)
        {
            Index1D copyIndex = new(_inputShape.Volume * batchSize);
            GPUManager.CopyAction(copyIndex, _buffers.Input, _inputCopy.GetArrayViewEmpty<float>());

            _buffers.Output.SubView(0, _outputUnits * batchSize).MemSetToZero();

            Index3D index = new(_inputShape.Volume, batchSize, _outputUnits);
            s_forwardAction(index, _buffers.Input, _buffers.Output, _filters.WeightsGPU<float>(), Info);

            Synchronize();

            _inputCopy.DecrementLiveCount();

            _filters.DecrementLiveWeights();


        }

        /// <inheritdoc/>
        public override void Reset()
        {

            float variance = 2f / (_outputUnits + _inputShape.Volume);
            float stdDev = MathF.Sqrt(variance);
            _filters.Reset(0, 0.02f);
        }

        /// <inheritdoc/>
        public override Shape Startup(Shape inputShapes, IOBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            BaseStartup(inputShapes, buffers);

            float variance = 2f / (_outputUnits + _inputShape.Volume);
            float stdDev = MathF.Sqrt(variance);

            _filters ??= new Weights(_inputShape.Dimensions * inputShapes.Area * _outputUnits, _weightInitializer, this);

            _inputCopy = new Vector(_inputShape.Dimensions * maxBatchSize * inputShapes.Area);

            return _outputShape;
        }

        private static void BackwardsFilterKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> input, ArrayView<float> filterGradient, LayerInfo info)
        {
            int inputArea = info.InputArea * info.InputDimensions;

            Atomic.Add(ref filterGradient[index.X + inputArea * index.Z], inGradient[index.Z + info.OutputArea * index.Y] * input[index.X + inputArea * index.Y]);
        }

        private static void BackwardsOutKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<float> filter, LayerInfo info)
        {
            int inputArea = info.InputArea * info.InputDimensions;

            Atomic.Add(ref outGradient[index.X + inputArea * index.Y], inGradient[index.Z + info.OutputArea * index.Y] * filter[index.X + inputArea * index.Z]);
        }

        private static void ForwardKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, ArrayView<float> filter, LayerInfo info)
        {
            int inputArea = info.InputArea * info.InputDimensions;

            Atomic.Add(ref output[index.Z + info.OutputArea * index.Y], input[index.X + inputArea * index.Y] * filter[index.X + inputArea * index.Z]);
        }

        protected override void BackwardsNoUpdate(int batchSize)
        {
            _buffers.OutGradient.SubView(0, batchSize * _inputShape.Volume).MemSetToZero();

            Index3D index = new(_inputShape.Volume, batchSize, _outputUnits);
            s_backwardsOutAction(index, _buffers.InGradient, _buffers.OutGradient, _filters.WeightsGPU<float>(), Info);

            Synchronize();

            _filters.DecrementLiveWeights();
        }

        protected override void BackwardsUpdate(int batchSize)
        {
            _buffers.OutGradient.SubView(0, batchSize * _inputShape.Volume).MemSetToZero();

            Index3D index = new(_inputShape.Volume, batchSize, _outputUnits);

            s_backwardsOutAction(index, _buffers.InGradient, _buffers.OutGradient, _filters.WeightsGPU<float>(), Info);
            s_backwardsFilterAction(index, _buffers.InGradient, _inputCopy.GetArrayView<float>(), _filters.GradientGPU<float>(), Info);

            Synchronize();

            _inputCopy.DecrementLiveCount();

            _filters.DecrementLiveWeights();
            _filters.DecrementLiveGradient();
            _filters.UpdateWeights(_adamHyperParameters);

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
