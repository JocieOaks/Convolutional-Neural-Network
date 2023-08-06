using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ConvolutionalNeuralNetwork.Layers.Initializers;
using ILGPU;
using ILGPU.Backends.EntryPoints;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Runtime.Serialization;


namespace ConvolutionalNeuralNetwork.Layers.Weighted
{
    public abstract class WeightedLayer : Layer
    {
        private bool UseBias => _bias != null;
        private Weights _bias;
        protected Weights _weights;

        public WeightedLayer(int filterSize, int stride, Weights weights, Weights bias) : base (filterSize, stride)
        {
            _weights = weights;
            _bias = bias;
        }


        private static void BiasKernal(Index3D index, ArrayView<float> value, ArrayView<float> bias, int dimensions, int length)
        {
            Atomic.Add(ref value[(index.Z * dimensions + index.Y) * length + index.X], bias[index.Y]);
        }

        private static void BiasGradientKernal(Index2D index, ArrayView<float> biasGradient, ArrayView<float> inGradient, int dimensions, int length)
        {
            float sum = 0;
            for (int i = 0; i < length; i++)
            {
                 sum += inGradient[(index.Y * dimensions + index.X) * length + i];
            }
            Atomic.Add(ref biasGradient[index.X], sum);
        }


        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, int, int> s_biasAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, int, int>(BiasKernal);

        private static readonly Action<Index2D, ArrayView<float>, ArrayView<float>, int, int> s_biasGradientAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, int, int>(BiasGradientKernal);
        
        [JsonConstructor] protected WeightedLayer() { }

        protected abstract void ForwardChild(int batchSize);
        protected abstract void BackwardsUpdate(int batchSize);
        protected abstract void BackwardsNoUpdate(int batchSize);

        public sealed override void Forward(int batchSize)
        {
            ForwardChild(batchSize);
            if (_bias != null)
            {
                Index3D biasIndex = new(_outputShape.Area, _outputShape.Dimensions, batchSize);
                s_biasAction(biasIndex, _buffers.Output, _bias.WeightsGPU<float>(), _outputShape.Dimensions, _outputShape.Area);

                Synchronize();

                _bias.DecrementLiveWeights();
            }
        }

        public sealed override void Backwards(int batchSize, bool update)
        {
            if (update)
            {
                BackwardsUpdate(batchSize);
                if (UseBias)
                {
                    Index2D biasIndex = new(_outputShape.Dimensions, batchSize);
                    s_biasGradientAction(biasIndex, _bias.GradientGPU<float>(), _buffers.InGradient, _outputShape.Dimensions, _outputShape.Area);

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
            if (UseBias)
            {
                (_bias as Weights).TestFilterGradient(this, input, output, _buffers, batchSize);
            }
        }

        protected abstract int WeightLength { get; }

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

            buffer.Allocate(batchSize);
            complimentBuffer.Allocate(batchSize);
            IOBuffers.SetCompliment(buffer, complimentBuffer);


            inputShape = new Shape(inputSize, inputSize, inputDimensions);

            return (inputShape, outputShape);
        }
    }
}
