using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers.Loss
{
    public abstract class Loss
    {
        protected PairedBuffers Buffers;
        protected TensorShape OutputShape;
        protected Vector Truth;
        protected Vector Losses = new(1);
        protected Vector Accuracy = new(1);

        public virtual void Startup(PairedBuffers buffers, TensorShape outputShape, int maxBatchSize)
        {
            Buffers = buffers;
            OutputShape = outputShape;
            Truth = new Vector(maxBatchSize * outputShape.Volume);
        }

        public abstract (float, float) GetLoss(Vector[] groundTruth);
    }
}
