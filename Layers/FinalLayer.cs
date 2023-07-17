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
        protected uint _batchSize;
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
        public abstract Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, uint batchSize);

        protected void BaseStartup(Shape[] inputShapes, IOBuffers buffers, uint batchSize)
        {
            _inputDimensions = inputShapes.Length;

            _batchSize = (uint)batchSize;
            _layerInfos = new ILayerInfo[_inputDimensions];
            _inputShapes = inputShapes;

            for (int i = 0; i < _inputDimensions; i++)
            {
                _layerInfos[i] = new LayerInfo()
                {
                    InputWidth = inputShapes[i].Width,
                    InputLength = inputShapes[i].Length,
                };
            }

            _outputShapes = new Shape[1];
            _outputShapes[0] = new Shape(1, _outputUnits);

            _buffers = buffers;
            buffers.OutputDimensionArea(0, _outputUnits);
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
    }
}
