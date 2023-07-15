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
        protected int _outputUnits;
        protected IOBuffers _buffers;
        protected int _inputDimensions;
        protected ILayerInfo[] _layerInfos;

        protected FinalLayer(int outputUnits)
        {
            _outputUnits = outputUnits;
        }

        [JsonConstructor] protected FinalLayer() { }

        public abstract string Name { get; }
        public abstract void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay);
        public abstract void Forward();
        public abstract void Reset();
        public abstract FeatureMap[,] Startup(FeatureMap[,] inputs, IOBuffers buffers);

        protected void BaseStartup(FeatureMap[,] inputs, IOBuffers buffers)
        {
            _inputDimensions = inputs.GetLength(0);

            _batchSize = (uint)inputs.GetLength(1);
            _layerInfos = new ILayerInfo[_inputDimensions];

            for (int i = 0; i < _inputDimensions; i++)
            {
                _layerInfos[i] = new StaticLayerInfo()
                {
                    Width = inputs[i, 0].Width,
                    Length = inputs[i, 0].Length,
                };
            }

            _buffers = buffers;
            buffers.OutputDimensionArea(0, _outputUnits);
        }

        protected void DecrementCacheabble(Cacheable[,] caches, uint decrement = 1)
        {
            for (int i = 0; i < caches.GetLength(0); i++)
            {
                for (int j = 0; j < caches.GetLength(1); j++)
                {
                    caches[i, j].DecrementLiveCount(decrement);
                }
            }
        }

        protected void Synchronize()
        {
            GPUManager.Accelerator.Synchronize();
        }
    }
}
