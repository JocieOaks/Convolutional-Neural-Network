using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers
{
    public abstract class FinalLayer : ILayer
    {
        protected int _batchSize;
        [JsonProperty] protected int _outputUnits;
        protected IOBuffers _buffers;
        protected int _inputDimensions;
        protected ILayerInfo _layerInfo;
        protected Shape _inputShape;
        protected Shape _outputShapes;

        protected FinalLayer(int outputUnits)
        {
            _outputUnits = outputUnits;
        }

        [JsonConstructor] protected FinalLayer() { }

        public abstract string Name { get; }
        public abstract void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay);
        public abstract void Forward();
        public abstract void Reset();
        public abstract Shape Startup(Shape inputShapes, IOBuffers buffers, int batchSize);

        protected void BaseStartup(Shape inputShapes, IOBuffers buffers, int batchSize)
        {
            _inputDimensions = inputShapes.Dimensions;

            _batchSize = batchSize;
            _inputShape = inputShapes;

            _layerInfo = new LayerInfo()
            {
                InputWidth = inputShapes.Width,
                InputLength = inputShapes.Length,
                OutputWidth = _outputUnits,
                OutputLength = 1,
                InputDimensions = _inputDimensions,
                OutputDimensions = 1
            };
            
            _outputShapes = new Shape(_outputUnits, 1, 1);

            _buffers = buffers;
            buffers.OutputDimensionArea(_outputUnits);
        }

        protected static void DecrementCacheabble(Cacheable[,] caches, uint decrement = 1)
        {
            for (int i = 0; i < caches.GetLength(0); i++)
            {
                for (int j = 0; j < caches.GetLength(1); j++)
                {
                    caches[i, j].DecrementLiveCount(decrement);
                }
            }
        }

        protected static void Synchronize()
        {
            GPUManager.Accelerator.Synchronize();
        }

        protected (Shape, Shape) FilterTestSetup(int inputDimensions, int batchSize, int inputSize)
        {
            Shape inputShape = new Shape(inputSize, inputSize, inputDimensions);


            IOBuffers buffer = new();
            IOBuffers complimentBuffer = new();
            complimentBuffer.OutputDimensionArea(inputDimensions * inputSize * inputSize);

            Shape outputShape = Startup(inputShape, buffer, batchSize);
            buffer.Allocate(batchSize);
            complimentBuffer.Allocate(batchSize);
            IOBuffers.SetCompliment(buffer, complimentBuffer);


            inputShape = new Shape(inputSize, inputSize, inputDimensions);

            return (inputShape, outputShape);
        }
    }
}
