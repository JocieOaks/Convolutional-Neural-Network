using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers.Loss
{
    public abstract class Loss
    {
        protected PairedGPUViews views;
        protected TensorShape OutputShape;
        protected Vector Truth;
        protected Vector Losses = new(1);
        protected Vector Accuracy = new(1);

        public virtual void Startup(PairedGPUViews views, TensorShape outputShape, int maxBatchSize)
        {
            this.views = views;
            OutputShape = outputShape;
            Truth = new Vector(maxBatchSize * outputShape.Volume);
        }

        public abstract (float, float) GetLoss(Vector[] groundTruth);
    }
}
