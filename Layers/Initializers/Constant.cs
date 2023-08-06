﻿using ConvolutionalNeuralNetwork.Layers.Serial;

namespace ConvolutionalNeuralNetwork.Layers.Initializers
{
    public class Constant : IWeightInitializer
    {
        float _constant;

        public Constant(float constant)
        {
            _constant = constant;
        }

        public float GetWeight(SerialWeighted layer)
        {
            return _constant;
        }
    }
}
