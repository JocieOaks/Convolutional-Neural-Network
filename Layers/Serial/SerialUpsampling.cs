using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialUpsampling : ISerial
    {
        public int Scale { get; init; }

        public SerialUpsampling(int scale)
        {
            Scale = scale;
        }

        [JsonConstructor] private SerialUpsampling() { }

        public Layer Construct()
        {
            return new Upsampling(Scale);
        }

        public Shape Initialize(Shape inputShape)
        {
            return new Shape(Scale * inputShape.Width, Scale * inputShape.Length, inputShape.Dimensions);
        }
    }
}
