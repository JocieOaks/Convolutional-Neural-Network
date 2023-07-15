using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    public class Dense : FinalLayer
    {
        [JsonProperty] private Weights[,] _filters;
        private FeatureMap[,] _inputs;

        private static readonly Action<Index2D, ArrayView<float>, VariableView<float>, ArrayView<float>> s_forwardAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, VariableView<float>, ArrayView<float>>(ForwardKernel);
        private static readonly Action<Index2D, VariableView<float>, ArrayView<float>, ArrayView<float>> s_backwardsOutAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, VariableView<float>, ArrayView<float>, ArrayView<float>>(BackwardsOutKernel);
        private static readonly Action<Index2D, VariableView<float>, ArrayView<float>, ArrayView<float>> s_backwardsFilterAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, VariableView<float>, ArrayView<float>, ArrayView<float>>(BackwardsFilterKernel);

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

            for (int i = 0; i < _outputUnits; i++)
            {
                for (int j = 0; j < _inputDimensions; j++)
                {
                    Index2D index = new(Infos(i).InputArea, 3);
                    for (int k = 0; k < _batchSize; k++)
                    {
                        s_backwardsOutAction(index, _buffers.FirstGradient(k).VariableView(j), _filters[i, j].WeightsGPU<float>(), _buffers.OutGradientsFloat[j % _inputDimensions, k]);
                        s_backwardsFilterAction(index, _buffers.FirstGradient(k).VariableView(j), _inputs[j, k].GetArrayView<float>(), _filters[i, j].GradientGPU<float>());
                    }
                }
            }

            Synchronize();
            DecrementCacheabble(_inputs, (uint)_outputUnits);

            for (int i = 0; i < _outputUnits; i++)
            {
                for (int j = 0; j < _inputDimensions; j++)
                {
                    _filters[i, j].DisposeWeights(_batchSize);
                    _filters[i, j].DisposeGradient(_batchSize);
                    _filters[i, j].UpdateWeights(learningRate, firstMomentDecay, secondMomentDecay);
                }
            }
        }

        private void BackwardsNoUpdate()
        {
            for (int i = 0; i < _outputUnits; i++)
            {
                for (int j = 0; j < _inputDimensions; j++)
                {
                    Index2D index = new(Infos(i).InputArea, 3);
                    for (int k = 0; k < _batchSize; k++)
                    {
                        s_backwardsOutAction(index, _buffers.FirstGradient(k).VariableView(j), _filters[i, j].WeightsGPU<float>(), _buffers.OutGradientsFloat[j % _inputDimensions, k]);
                    }
                }
            }
            Synchronize();

            for (int i = 0; i < _outputUnits; i++)
            {
                for (int j = 0; j < _inputDimensions; j++)
                {
                    _filters[i, j].DisposeWeights(_batchSize);
                }
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index1D index = new(Infos(i).InputArea);
                for (int j = 0; j < _batchSize; j++)
                {
                    GPUManager.CopyAction(index, _buffers.InputsColor[i, j], _inputs[i, j].GetArrayViewEmpty<Color>());
                }
            }

            for (int i = 0; i < _outputUnits; i++)
            {
                for (int j = 0; j < _inputDimensions; j++)
                {
                    Index2D index = new(Infos(i).InputArea, 3);
                    for (int k = 0; k < _batchSize; k++)
                    {
                        s_forwardAction(index, _buffers.InputsFloat[j, k], _buffers.FinalOutput(k).VariableView(i), _filters[i, j].WeightsGPU<float>());
                    }
                }
            }

            Synchronize();

            DecrementCacheabble(_inputs);

            for (int i = 0; i < _outputUnits; i++)
            {
                for (int j = 0; j < _inputDimensions; j++)
                {
                    _filters[i,j].DisposeWeights(_batchSize);
                }
            }
        }

        /// <inheritdoc/>
        public override void Reset()
        {
            for (int i = 0; i < _outputUnits; i++)
            {
                for (int j = 0; j < _inputDimensions; j++)
                {
                    float variance = 2f / (3 * Infos(j).InputArea * _batchSize + 1);
                    float stdDev = MathF.Sqrt(variance);
                    _filters[i, j] = new Weights(3 * Infos(j).InputArea, 0, stdDev);
                }
            }
        }

        /// <inheritdoc/>
        public override FeatureMap[,] Startup(FeatureMap[,] inputs, IOBuffers buffers)
        {
            BaseStartup(inputs, buffers);
            for (int i = 0; i < _outputUnits; i++)
            {
                for (int j = 0; j < _inputDimensions; j++)
                {
                    if (_filters[i, j] == null)
                    {
                        float variance = 2f / (3 * Infos(j).InputArea * _batchSize + 1);
                        float stdDev = MathF.Sqrt(variance);
                        _filters[i, j] = new Weights(3 * Infos(j).InputArea, 0, stdDev);
                    }
                }
            }

            _inputs = inputs;

            return null;
        }

        private static void ForwardKernel(Index2D index, ArrayView<float> input, VariableView<float> output, ArrayView<float> filter)
        {
            int arrayIndex = index.X * 3 + index.Y;
            Atomic.Add(ref output.Value, input[arrayIndex] * filter[arrayIndex]);
        }

        private static void BackwardsOutKernel(Index2D index, VariableView<float> inGradient, ArrayView<float> outGradient, ArrayView<float> filter)
        {
            int arrayIndex = index.X * 3 + index.Y;
            outGradient[arrayIndex] = inGradient.Value * filter[arrayIndex];
        }

        private static void BackwardsFilterKernel(Index2D index, VariableView<float> inGradient, ArrayView<float> input, ArrayView<float> filterGradient)
        {
            int arrayIndex = index.X * 3 + index.Y;
            filterGradient[arrayIndex] = inGradient.Value * input[arrayIndex];
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
