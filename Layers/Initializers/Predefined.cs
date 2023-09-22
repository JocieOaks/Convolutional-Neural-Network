using ConvolutionalNeuralNetwork.Layers.Serial;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers.Initializers
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
