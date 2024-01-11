using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    public class Reshape : Layer
    { 
        public Reshape(TensorShape outputShape)
        {
            OutputShape = outputShape;
        }

        public override string Name => "Reshape Layer";

        public override void Backwards(int batchSize, bool update)
        {
        }

        public override void Forward(int batchSize)
        {
        }

        /// <inheritdoc />
        [JsonIgnore] public override bool Reflexive => true;

        public override TensorShape Startup(TensorShape inputShape, PairedBuffers buffers, int maxBatchSize)
        {
            if (Ready)
                return OutputShape;
            Ready = true;

            int inputLength = inputShape.Volume;

            int outputLength = OutputShape.Volume;

            if(inputLength != outputLength)
            {
                throw new ArgumentException("Input and output shapes have different lengths.");
            }

            return OutputShape;
        }
    }
}
