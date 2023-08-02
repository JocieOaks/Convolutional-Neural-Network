using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ConvolutionalNeuralNetwork.Layers.Initializers;
using ILGPU;
using ILGPU.Backends.EntryPoints;
using Newtonsoft.Json;
using System.Runtime.Serialization;


namespace ConvolutionalNeuralNetwork.Layers.Weighted
{
    public abstract class WeightedLayer : Layer
    {
        private IWeightInitializer _weightInitializer;
        private bool _useBias;
        private IWeights _bias;
        protected IWeights _weights;

        public WeightedLayer(int filterSize, int stride, IWeightInitializer weightInitializer, bool useBias) : base(filterSize, stride)
        {
            _weightInitializer = weightInitializer;
            _useBias = useBias;
        }

        public WeightedLayer(int filterSize, int stride, SharedWeights weights, SharedWeights bias) : base (filterSize, stride)
        {
            _weights = weights;
            if(bias != null)
            {
                _useBias = true;
                _bias = bias;
            }
        }

        [JsonConstructor] protected WeightedLayer() { }

        protected abstract void ForwardChild(int batchSize);
        protected abstract void BackwardsUpdate(int batchSize);
        protected abstract void BackwardsNoUpdate(int batchSize);

        public sealed override void Forward(int batchSize)
        {
            ForwardChild(batchSize);
            if (_bias != null)
            {
                Index3D biasIndex = new(batchSize, _outputShape.Dimensions, _outputShape.Area);
                GPUManager.BiasAction(biasIndex, _buffers.Output, _bias.WeightsGPU<float>(), _outputShape.Dimensions, _outputShape.Area);

                Synchronize();

                _bias.DecrementLiveWeights();
            }
        }

        public sealed override void Backwards(int batchSize, bool update)
        {
            if (update)
            {
                BackwardsUpdate(batchSize);
                if (_useBias)
                {
                    Index3D biasIndex = new(batchSize, _outputShape.Dimensions, _outputShape.Area);
                    GPUManager.BiasGradientAction(biasIndex, _bias.GradientGPU<float>(), _buffers.InGradient, _outputShape.Dimensions, _outputShape.Area);

                    Synchronize();

                    _bias.DecrementLiveGradient();
                }
            }
            else
            {
                BackwardsNoUpdate(batchSize);
            }
        }

        protected void BiasTest(Shape input, Shape output, int batchSize)
        {
            if (_useBias)
            {
                (_bias as Weights).TestFilterGradient(this, input, output, _buffers, batchSize);
            }
        }

        public IEnumerable<Weights> SetUpWeights()
        {
            if(_weights is SharedWeights shared)
            {
                if (shared.Initialized == false)
                {
                    Weights weights = new Weights(WeightLength, shared.WeightInitializer, this);
                    shared.SetWeights(weights);
                    yield return weights;
                }
                else
                {
                    if(_weights.Length != WeightLength)
                    {
                        throw new Exception("Shared Weights are incorrect size for this layer.");
                    }
                }
            }
            else
            {
                _weights ??= new Weights(WeightLength, _weightInitializer, this);
                yield return _weights as Weights;
            }
            
            if (_useBias)
            {
                if (_bias is SharedWeights sharedBias)
                {
                    if (sharedBias.Initialized == false)
                    {
                        Weights bias = new(_outputShape.Dimensions);
                        sharedBias.SetWeights(bias);
                        yield return bias;
                    }
                    else
                    {
                        if (_bias.Length != _outputShape.Dimensions)
                        {
                            throw new Exception("Shared Weights are incorrect size for this layer.");
                        }
                    }
                }
                else
                {
                    _bias ??= new Weights(_outputShape.Dimensions);
                    yield return _bias as Weights;
                }
            }
        }

        protected abstract int WeightLength { get; }

        [OnDeserialized]
        public void OnDeserialized(StreamingContext context)
        {
            _useBias = _bias != null;
        }

        public int FanIn => _inputShape.Volume;

        public int FanOut => _outputShape.Volume;

        protected (Shape, Shape) FilterTestSetup(int inputDimensions, int batchSize, int inputSize)
        {
            Shape inputShape = new Shape(inputSize, inputSize, inputDimensions);


            IOBuffers buffer = new();
            IOBuffers complimentBuffer = new();
            complimentBuffer.OutputDimensionArea(inputDimensions * inputSize * inputSize);

            Shape outputShape = Startup(inputShape, buffer, batchSize);
            var adam = new AdamHyperParameters()
            {
                LearningRate = 0
            };
            adam.Update();
            SetUpWeights();

            buffer.Allocate(batchSize);
            complimentBuffer.Allocate(batchSize);
            IOBuffers.SetCompliment(buffer, complimentBuffer);


            inputShape = new Shape(inputSize, inputSize, inputDimensions);

            return (inputShape, outputShape);
        }
    }
}
