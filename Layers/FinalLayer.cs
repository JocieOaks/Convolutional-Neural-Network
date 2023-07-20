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
        protected ILayerInfo[] _layerInfos;
        protected Shape[] _inputShapes;
        protected Shape[] _outputShapes;

        protected FinalLayer(int outputUnits)
        {
            _outputUnits = outputUnits;
        }

        [JsonConstructor] protected FinalLayer() { }

        public abstract string Name { get; }
        public abstract void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay);
        public abstract void Forward();
        public abstract void Reset();
        public abstract Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, int batchSize);

        protected void BaseStartup(Shape[] inputShapes, IOBuffers buffers, int batchSize)
        {
            _inputDimensions = inputShapes.Length;

            _batchSize = batchSize;
            _layerInfos = new ILayerInfo[_inputDimensions];
            _inputShapes = inputShapes;

            for (int i = 0; i < _inputDimensions; i++)
            {
                _layerInfos[i] = new LayerInfo()
                {
                    InputWidth = inputShapes[i].Width,
                    InputLength = inputShapes[i].Length,
                    OutputWidth = _outputUnits,
                    OutputLength = 1,
                    InputDimensions = _inputDimensions,
                    OutputDimensions = 1
                };
            }

            _outputShapes = new Shape[1];
            _outputShapes[0] = new Shape(1, _outputUnits);

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

        protected (Shape[], Shape[]) FilterTestSetup(int dimensionMultiplier, int batchSize)
        {
            int outputDimensions, inputDimensions;
            if (dimensionMultiplier >= 1)
            {
                inputDimensions = 1;
                outputDimensions = dimensionMultiplier;
            }
            else
            {
                inputDimensions = -dimensionMultiplier;
                outputDimensions = 1;
            }
            Shape[] inputShape = new Shape[inputDimensions];
            for (int i = 0; i < inputDimensions; i++)
            {
                inputShape[i] = new Shape(3, 3);
            }

            IOBuffers buffer = new();
            IOBuffers complimentBuffer = new();
            complimentBuffer.OutputDimensionArea(inputDimensions * 9);

            Startup(inputShape, buffer, batchSize);
            buffer.Allocate(batchSize);
            complimentBuffer.Allocate(batchSize);
            IOBuffers.SetCompliment(buffer, complimentBuffer);

            inputShape = new Shape[inputDimensions * batchSize];
            for (int i = 0; i < inputDimensions * batchSize; i++)
            {
                inputShape[i] = new Shape(3, 3);
            }

            Shape[] outputShape = new Shape[outputDimensions * batchSize];
            for (int i = 0; i < outputDimensions * batchSize; i++)
            {
                outputShape[i] = _outputShapes[0];
            }

            return (inputShape, outputShape);
        }
    }
}
