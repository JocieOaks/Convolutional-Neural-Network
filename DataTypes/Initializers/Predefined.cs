using ConvolutionalNeuralNetwork.Layers.Serial;

namespace ConvolutionalNeuralNetwork.DataTypes.Initializers
{
    public class Predefined : IWeightInitializer
    {
        private List<float> _weights;
        int _step = 0;
        public Predefined(List<float> weights)
        {
            _weights = weights;
        }
        public float GetWeight(SerialWeighted layer)
        {
            return _weights[_step++ % _weights.Count];
        }
    }
}
