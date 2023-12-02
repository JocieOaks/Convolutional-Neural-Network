using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers.Loss
{
    public abstract class Loss
    {
        protected IOBuffers Buffers;
        protected Shape OutputShape;
        protected Vector Truth;
        protected Vector Losses = new(1);
        protected Vector Accuracy = new(1);

        public virtual void Startup(IOBuffers buffers, Shape outputShape, int maxBatchSize)
        {
            Buffers = buffers;
            OutputShape = outputShape;
            Truth = new Vector(maxBatchSize * outputShape.Volume);
        }

        public abstract (float, float) GetLoss(Vector[] groundTruth);
    }
}
