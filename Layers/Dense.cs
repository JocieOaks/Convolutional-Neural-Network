using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Collections.Generic;

namespace ConvolutionalNeuralNetwork.Layers
{
    public class Dense : FinalLayer, IPrimaryLayer
    {
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> s_backwardsFilterAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(BackwardsFilterKernel);

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> s_backwardsOutAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(BackwardsOutKernel);

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> s_forwardAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(ForwardKernel);

        static bool temp = true;
        [JsonProperty] private Weights _bias;
        [JsonProperty] private Weights _filters;
        private Vector _inputCopy;
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

        /// <summary>
        /// Gets the <see cref="LayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="LayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="LayerInfo"/> corresponding to an input dimension.</returns>
        private LayerInfo Info => (LayerInfo)_layerInfo;

        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            if (learningRate <= 0)
                BackwardsNoUpdate();
            else
                BackwardsUpdate(learningRate, firstMomentDecay, secondMomentDecay);
        }

        public void FilterTest(int inputDimensions, int batchSize, int inputSize)
        {
            (Shape input, Shape output) = FilterTestSetup(inputDimensions, batchSize, inputSize);

            _filters.TestFilterGradient(this, input, output, _buffers, batchSize);
            _bias.TestFilterGradient(this, input, output, _buffers, batchSize);

        }

        /// <inheritdoc/>
        public override void Forward()
        {
            Index1D copyIndex = new(_inputDimensions * _inputShape.Area * _batchSize);
            GPUManager.CopyAction(copyIndex, _buffers.Input, _inputCopy.GetArrayViewEmpty<float>());

            _buffers.Output.SubView(0, _outputUnits * _batchSize).MemSetToZero();

            Index3D index = new(_inputDimensions * _inputShape.Area, _batchSize, _outputUnits);
            s_forwardAction(index, _buffers.Input, _buffers.Output, _filters.WeightsGPU<float>(), Info);

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

            float variance = 2f / (_outputUnits + _inputDimensions * _inputShape.Area);
            float stdDev = MathF.Sqrt(variance);
            _filters.Reset(0, 0.02f);

            _bias.Reset(0);
        }

        /// <inheritdoc/>
        public override Shape Startup(Shape inputShapes, IOBuffers buffers, int batchSize)
        {
            BaseStartup(inputShapes, buffers, batchSize);

            float variance = 2f / (_outputUnits + _inputDimensions * _inputShape.Area);
            float stdDev = MathF.Sqrt(variance);

            if (temp)
            {
                _filters ??= new Weights(_inputDimensions * inputShapes.Area * _outputUnits, 0, 0.02f);
                temp = false;
            }
            else
            {
                float limit = MathF.Sqrt(6f / (_outputUnits + _inputDimensions * inputShapes.Area));
                _filters ??= new Weights(_inputDimensions * inputShapes.Area * _outputUnits, limit, true);
            }
            _bias ??= new Weights(_outputUnits, 0);

            _inputCopy = new Vector(_inputDimensions * _batchSize * inputShapes.Area);

            return _outputShapes;
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

        private void BackwardsNoUpdate()
        {
            _buffers.OutGradient.SubView(0, _batchSize * _inputDimensions * _inputShape.Area).MemSetToZero();

            Index3D index = new(_inputDimensions * _inputShape.Area, _batchSize, _outputUnits);
            s_backwardsOutAction(index, _buffers.InGradient, _buffers.OutGradient, _filters.WeightsGPU<float>(), Info);

            Synchronize();

            _filters.DecrementLiveWeights();
        }

        private void BackwardsUpdate(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            _buffers.OutGradient.SubView(0, _batchSize * _inputDimensions * _inputShape.Area).MemSetToZero();

            Index3D index = new(_inputDimensions * _inputShape.Area, _batchSize, _outputUnits);

            s_backwardsOutAction(index, _buffers.InGradient, _buffers.OutGradient, _filters.WeightsGPU<float>(), Info);
            s_backwardsFilterAction(index, _buffers.InGradient, _inputCopy.GetArrayView<float>(), _filters.GradientGPU<float>(), Info);

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
    }
}
