using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialInput : ISerial
    {
        [JsonProperty] Shape _inputShape;

        public SerialInput(Shape inputShape)
        {
            _inputShape = inputShape;
        }

        [JsonConstructor] public SerialInput() { }

        public Layer Construct()
        {
            return new InputLayer(_inputShape);
        }

        public Shape Initialize(Shape inputShape)
        {
            return _inputShape;
        }
    }
}
