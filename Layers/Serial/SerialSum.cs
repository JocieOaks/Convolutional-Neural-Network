using ConvolutionalNeuralNetwork.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialSum : ISerial
    {
        public int OutputDimensions { get; init; }

        public SerialSum(int outputDimensions)
        {
            OutputDimensions = outputDimensions;
        }

        [JsonConstructor] private SerialSum() { }

        public Layer Construct()
        {
            return new Summation(OutputDimensions);
        }

        public Shape Initialize(Shape inputShape)
        {
            if(inputShape.Dimensions %  OutputDimensions != 0)
            {
                throw new ArgumentException("Input cannot be summed evenly over output dimensions.");
            }

            return new Shape(inputShape.Width, inputShape.Length, OutputDimensions);
        }
    }
}
