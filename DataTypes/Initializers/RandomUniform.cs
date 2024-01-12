using ConvolutionalNeuralNetwork.Layers.Serial;

namespace ConvolutionalNeuralNetwork.DataTypes.Initializers
{
    public class RandomUniform : IWeightInitializer
    {
        float _max;
        float _min;
        float _delta;
        public RandomUniform(float max, float min = float.NaN)
        {
            _max = max;
            if (float.IsNaN(min))
            {
                _min = -max;
            }
            else
            {
                _min = min;
            }
            if (max < min)
            {
                throw new ArgumentException("Max is less than min.");
            }

            _delta = _max - _min;
        }

        public float GetWeight(SerialWeighted layer)
        {
            return Utility.Random.NextSingle() * _delta + _min;
        }
    }
}
