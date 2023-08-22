using ConvolutionalNeuralNetwork.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers.Loss
{
    public abstract class Loss
    {
        protected IOBuffers _buffers;
        protected Shape _outputShape;
        protected Vector _truth;
        protected Vector _loss = new(1);
        protected Vector _accuracy = new(1);

        public virtual void Startup(IOBuffers buffers, Shape outputShape, int maxBatchSize)
        {
            _buffers = buffers;
            _outputShape = outputShape;
            _truth = new Vector(maxBatchSize * outputShape.Volume);
        }

        public abstract (float, float) GetLoss(Vector[] groundTruth);
    }
}
