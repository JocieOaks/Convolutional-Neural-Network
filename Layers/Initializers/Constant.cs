using ConvolutionalNeuralNetwork.Layers.Weighted;

namespace ConvolutionalNeuralNetwork.Layers.Initializers
{
    public class Constant : IWeightInitializer
    {
        float _constant;

        public Constant(float constant)
        {
            _constant = constant;
        }

        public float GetWeight(WeightedLayer layer)
        {
            return _constant;
        }
    }
}
