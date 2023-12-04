﻿using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialAvgPool : ISerial
    {
        public int Scale { get; init; }

        public SerialAvgPool(int scale)
        {
            Scale = scale;
        }

        [JsonConstructor] private SerialAvgPool() { }

        public Layer Construct()
        {
            return new AveragePool(Scale);
        }

        public TensorShape Initialize(TensorShape inputShape)
        {
            return new TensorShape(inputShape.Width / Scale, inputShape.Length / Scale, inputShape.Dimensions);
        }
    }
}
