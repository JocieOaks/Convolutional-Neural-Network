using ConvolutionalNeuralNetwork.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public interface ISerial
    {
        Shape Initialize(Shape inputShape);

        Layer Construct();
    }
}
