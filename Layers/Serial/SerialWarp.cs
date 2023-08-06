using ConvolutionalNeuralNetwork.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialWarp : ISerial
    {
        public Layer Construct()
        {
            return new Warp();
        }

        public Shape Initialize(Shape inputShape)
        {
            return new Shape(inputShape.Width, inputShape.Length, inputShape.Dimensions - 2);
        }
    }
}
