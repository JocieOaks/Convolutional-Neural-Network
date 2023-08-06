using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.SkipConnection;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialConcatenate : ISerial
    {
        [JsonProperty] private readonly int _id;
        private readonly SerialFork _source;

        public SerialConcatenate(SerialFork source)
        {
            _source = source;
            _id = source.ID;
        }

        [JsonConstructor] SerialConcatenate() { }

        public Layer Construct()
        {
            if (SerialFork.Forks.TryGetValue(_id, out var split))
            {
                return split.GetConcatenationLayer();
            }
            else
            {
                throw new Exception("Skip connection fork cannot be found.");
            }
        }

        public Shape Initialize(Shape inputShape)
        {
            if (inputShape.Area != _source.OutputShape.Area)
            {
                throw new ArgumentException("Input shapes do not match.");
            }

            return new Shape(inputShape.Width, inputShape.Length, inputShape.Dimensions + _source.OutputShape.Dimensions);
        }
    }
}
