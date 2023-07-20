using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    public class Dense : FinalLayer, IPrimaryLayer
    {
        [JsonProperty] private Weights _filters;
        private Vector _inputCopy;
        [JsonProperty] private Weights _bias;
        private ArrayView<LayerInfo> _deviceInfos;

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>> s_forwardAction = 
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>>(ForwardKernel);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>> s_backwardsOutAction = 
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsOutKernel);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>> s_backwardsFilterAction = 
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsFilterKernel);

        /// <summary>
        /// Initializes a new instance of the <see cref="Dense"/> class.
        /// </summary>
        /// <param name="outputUnits">The number of units to output when performing a forward pass with <see cref="Dense"/>.</param>
        public Dense(int outputUnits) : base(outputUnits)
        {
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor] private Dense() { }

        /// <inheritdoc/>
        public override string Name => "Dense Layer";

        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            if (learningRate <= 0)
                BackwardsNoUpdate();
            else
                BackwardsUpdate(learningRate, firstMomentDecay, secondMomentDecay);
        }

        private void BackwardsUpdate(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            _buffers.OutGradient.SubView(0, _batchSize * _inputDimensions * Infos(0).InputArea).MemSetToZero();

            Index3D index = new(_batchSize, _inputDimensions * Infos(0).InputArea, _outputUnits);

            s_backwardsOutAction(index, _buffers.InGradient, _buffers.OutGradient, _filters.WeightsGPU<float>(), _deviceInfos);
            s_backwardsFilterAction(index, _buffers.InGradient, _inputCopy.GetArrayView<float>(), _filters.GradientGPU<float>(), _deviceInfos);

            Index3D biasIndex = new(_outputUnits, 1, _batchSize);
            GPUManager.AddAction(biasIndex, _bias.GradientGPU<float>(), _buffers.InGradient, _outputUnits);

            Synchronize();

            _inputCopy.DecrementLiveCount();
            _bias.DecrementLiveGradient();
            _bias.UpdateWeights(learningRate, firstMomentDecay, secondMomentDecay);

            _filters.DecrementLiveWeights();
            _filters.DecrementLiveGradient();
            _filters.UpdateWeights(learningRate, firstMomentDecay, secondMomentDecay);
            
        }

        private void BackwardsNoUpdate()
        {
            _buffers.OutGradient.SubView(0, _batchSize * _inputDimensions * Infos(0).InputArea).MemSetToZero();

            Index3D index = new(_batchSize, _inputDimensions * Infos(0).InputArea, _outputUnits);
            s_backwardsOutAction(index, _buffers.InGradient, _buffers.OutGradient, _filters.WeightsGPU<float>(), _deviceInfos);

            Synchronize();

            _filters.DecrementLiveWeights();
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            Index1D copyIndex = new(_inputDimensions * Infos(0).InputArea * _batchSize);
            GPUManager.CopyAction(copyIndex, _buffers.Input, _inputCopy.GetArrayViewEmpty<float>());

            _buffers.Output.SubView(0, _outputUnits * _batchSize).MemSetToZero();

            Index3D index = new(_batchSize, _inputDimensions * Infos(0).InputArea, _outputUnits);
            s_forwardAction(index, _buffers.Input, _buffers.Output, _filters.WeightsGPU<float>(), _deviceInfos);

            Index3D biasIndex = new(_outputUnits, _batchSize, 1);
            GPUManager.AddAction(biasIndex, _buffers.Output, _bias.WeightsGPU<float>(), _outputUnits);

            Synchronize();

            _bias.DecrementLiveWeights();

            _inputCopy.DecrementLiveCount();

            _filters.DecrementLiveWeights();
            
            
        }

        /// <inheritdoc/>
        public override void Reset()
        {

            float variance = 2f / (Infos(0).InputArea * _batchSize + 1);
            float stdDev = MathF.Sqrt(variance);
            _filters.Reset(0, 0.02f);
            
            _bias.Reset(0);
        }

        static bool temp = true;

        /// <inheritdoc/>
        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, int batchSize)
        {
            BaseStartup(inputShapes, buffers, batchSize);

            float variance = 2f / (_outputUnits + _inputDimensions * Infos(0).InputArea);
            float stdDev = MathF.Sqrt(variance);

            float limit = MathF.Sqrt(6f / (_outputUnits + _inputDimensions * Infos(0).InputArea));
            if (temp)
            {
                _filters ??= new Weights(_inputDimensions * inputShapes[0].Area * _outputUnits, 0, 0.02f);
            }
            else
            {
                _filters ??= new Weights(_inputDimensions * inputShapes[0].Area * _outputUnits, limit, true);
            }
            _bias ??= new Weights(_outputUnits, 0);

            _inputCopy = new Vector(_inputDimensions * _batchSize * inputShapes[0].Area);
            _deviceInfos = GPUManager.Accelerator.Allocate1D(Array.ConvertAll(_layerInfos, info => (LayerInfo)info)).View;

            return _outputShapes;
        }

        public void FilterTest(int outputMultiplier, int batchSize)
        {
            (Shape[] input, Shape[] output) = FilterTestSetup(outputMultiplier, batchSize);

            _filters.TestFilterGradient(this, input, output, _buffers);
            _bias.TestFilterGradient(this, input, output, _buffers);
            
        }

        private static void ForwardKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, ArrayView<float> filter, ArrayView<LayerInfo> infoView)
        {
            LayerInfo info = infoView[0];
            int inputArea = info.InputArea * info.InputDimensions;

            Atomic.Add(ref output[index.Z + info.OutputArea * index.X], input[index.Y + inputArea * index.X] * filter[index.Y + inputArea * index.Z]);
        }

        private static void BackwardsOutKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<float> filter, ArrayView<LayerInfo> infoView)
        {
            LayerInfo info = infoView[0];
            int inputArea = info.InputArea * info.InputDimensions;

            Atomic.Add(ref outGradient[index.Y + inputArea * index.X], inGradient[index.Z + info.OutputArea * index.X] * filter[index.Y + inputArea * index.Z]);
        }

        private static void BackwardsFilterKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> input, ArrayView<float> filterGradient, ArrayView<LayerInfo> infoView)
        {
            LayerInfo info = infoView[0];
            int inputArea = info.InputArea * info.InputDimensions;

            Atomic.Add(ref filterGradient[index.Y + inputArea * index.Z], inGradient[index.Z + info.OutputArea * index.X] * input[index.Y + inputArea * index.X]);
        }

        /// <summary>
        /// Gets the <see cref="LayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="LayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="LayerInfo"/> corresponding to an input dimension.</returns>
        private LayerInfo Infos(int index)
        {
            return (LayerInfo)_layerInfos[index % _inputDimensions];
        }
    }
}
