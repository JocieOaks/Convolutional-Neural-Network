using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    public class Dense : FinalLayer, IPrimaryLayer
    {
        [JsonProperty] private Weights[] _filters;
        private FeatureMap[,] _inputs;
        [JsonProperty] private Weights _bias;

        private static readonly Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>> s_forwardAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(ForwardKernel);
        private static readonly Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>> s_backwardsOutAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(BackwardsOutKernel);
        private static readonly Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>> s_backwardsFilterAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(BackwardsFilterKernel);

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
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index2D index = new(Infos(i).InputArea, _outputUnits);
                for (int j = 0; j < _batchSize; j++)
                {
                    s_backwardsOutAction(index, _buffers.FirstGradient(j), _buffers.OutGradientsFloat[i, j].SubView(0, Infos(i).InputArea), _filters[i].WeightsGPU<float>());
                    s_backwardsFilterAction(index, _buffers.FirstGradient(j), _inputs[i, j].GetArrayView<float>().SubView(0, Infos(i).InputArea), _filters[i].GradientGPU<float>());
                }
            }

            Synchronize();

            Index1D biasIndex = new(_outputUnits);
            for (int i = 0; i < _batchSize; i++)
            {
                GPUManager.AddAction(biasIndex, _bias.GradientGPU<float>(), _buffers.FirstGradient(i));
            }

            Synchronize();

            DecrementCacheabble(_inputs, 1);
            _bias.DisposeGradient(_batchSize);
            _bias.UpdateWeights(learningRate, firstMomentDecay, secondMomentDecay);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _filters[i].DisposeWeights(_batchSize);
                _filters[i].DisposeGradient(_batchSize);
                _filters[i].UpdateWeights(learningRate, firstMomentDecay, secondMomentDecay);
            }
        }

        private void BackwardsNoUpdate()
        {

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index2D index = new(Infos(i).InputArea, _outputUnits);
                for (int j = 0; j < _batchSize; j++)
                {
                    s_backwardsOutAction(index, _buffers.FirstGradient(j), _buffers.OutGradientsFloat[i, j], _filters[i].WeightsGPU<float>());
                }
            }
            
            Synchronize();

            for (int j = 0; j < _inputDimensions; j++)
            {
                _filters[j].DisposeWeights(_batchSize);
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
                    GPUManager.CopyAction(index, _buffers.InputsFloat[i, j], _inputs[i, j].GetArrayViewEmpty<float>());
                }
            }

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index2D index = new(Infos(i).InputArea, _outputUnits);
                for (int j = 0; j < _batchSize; j++)
                {
                    s_forwardAction(index, _buffers.InputsFloat[i, j].SubView(0, Infos(i).InputArea), _buffers.FinalOutput(j), _filters[i].WeightsGPU<float>());
                }
            }
            

            Synchronize();

            Index1D biasIndex = new(_outputUnits);
            for(int i = 0; i < _batchSize; i++)
            {
                GPUManager.AddAction(biasIndex, _buffers.FinalOutput(i), _bias.WeightsGPU<float>());
            }

            Synchronize();

            _bias.DisposeWeights(_batchSize);
            

            DecrementCacheabble(_inputs);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _filters[i].DisposeWeights(_batchSize);
            }
            
        }

        /// <inheritdoc/>
        public override void Reset()
        {
            for (int j = 0; j < _inputDimensions; j++)
            {
                float variance = 2f / (Infos(j).InputArea * _batchSize + 1);
                float stdDev = MathF.Sqrt(variance);
                _filters[j].Reset(0, 0.02f);
            }
            
            _bias.Reset(0);
        }

        /// <inheritdoc/>
        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, uint batchSize)
        {
            BaseStartup(inputShapes, buffers, batchSize);
            
            _filters ??= new Weights[_inputDimensions];


            for (int j = 0; j < _inputDimensions; j++)
            {
                if (_filters[j] == null)
                {
                    float variance = 2f / (_outputUnits + _inputDimensions * Infos(j).InputArea);
                    float stdDev = MathF.Sqrt(variance);
                    _filters[j] = new Weights(Infos(j).InputArea * _outputUnits, 0, stdDev);
                }
            }
            

            _bias ??= new Weights(_outputUnits, 0);

            _inputs = new FeatureMap[_inputDimensions, batchSize];
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < batchSize; j++)
                {
                    _inputs[i, j] = new FeatureMap(inputShapes[i]);
                }
            }

            return _outputShapes;
        }

        private static void ForwardKernel(Index2D index, ArrayView<float> input, ArrayView<float> output, ArrayView<float> filter)
        {

            Atomic.Add(ref output[index.Y], input[index.X] * filter[index.X + input.Length * index.Y]);
        }

        private static void BackwardsOutKernel(Index2D index, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<float> filter)
        {
            outGradient[index.X] = inGradient[index.Y] * filter[index.X + outGradient.Length * index.Y];
        }

        private static void BackwardsFilterKernel(Index2D index, ArrayView<float> inGradient, ArrayView<float> input, ArrayView<float> filterGradient)
        {
            Atomic.Add(ref filterGradient[index.X + input.Length * index.Y], inGradient[index.Y] * input[index.X]);
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
