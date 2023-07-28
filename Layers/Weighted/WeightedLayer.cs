using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ConvolutionalNeuralNetwork.Layers.Initializers;
using ILGPU;
using Newtonsoft.Json;
using System.Runtime.Serialization;


namespace ConvolutionalNeuralNetwork.Layers.Weighted
{
    public abstract class WeightedLayer : Layer
    {
        protected IWeightInitializer _weightInitializer;
        protected AdamHyperParameters _adamHyperParameters;
        private bool _useBias;
        [JsonProperty] private Weights _bias;

        public WeightedLayer(int filterSize, int stride, IWeightInitializer weightInitializer, bool useBias) : base(filterSize, stride)
        {
            _weightInitializer = weightInitializer;
            _useBias = useBias;
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

        public sealed override void Backwards(int batchSize)
        {
            if (!_adamHyperParameters.UpdateWeights)
            {
                BackwardsNoUpdate(batchSize);
            }
            else
            {
                BackwardsUpdate(batchSize);
                if (_useBias)
                {
                    Index3D biasIndex = new(batchSize, _outputShape.Dimensions, _outputShape.Area);
                    GPUManager.BiasGradientAction(biasIndex, _bias.GradientGPU<float>(), _buffers.InGradient, _outputShape.Dimensions, _outputShape.Area);

                    Synchronize();

                    _bias.DecrementLiveGradient();
                    _bias.UpdateWeights(_adamHyperParameters);
                }
            }
        }

        protected void BiasTest(Shape input, Shape output, int batchSize)
        {
            if (_useBias)
            {
                _bias.TestFilterGradient(this, input, output, _buffers, batchSize);
            }
        }

        public void SetUpWeights(AdamHyperParameters hyperParameters)
        {
            _adamHyperParameters = hyperParameters;
            if (_useBias)
            {
                _bias ??= new Weights(_outputShape.Dimensions);
            }
        }

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
            SetUpWeights(adam);

            buffer.Allocate(batchSize);
            complimentBuffer.Allocate(batchSize);
            IOBuffers.SetCompliment(buffer, complimentBuffer);


            inputShape = new Shape(inputSize, inputSize, inputDimensions);

            return (inputShape, outputShape);
        }
    }
}
