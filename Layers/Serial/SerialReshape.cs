using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialReshape : ISerial
    {
        [JsonProperty] private Shape _outputShape;

        public SerialReshape(Shape outputShape)
        {
            _outputShape = outputShape;
        }

        [JsonConstructor] private SerialReshape() { }

        public Layer Construct()
        {
            return new Reshape(_outputShape);
        }

        public Shape Initialize(Shape inputShape)
        {
            int inputLength = inputShape.Volume;

            int outputLength = _outputShape.Volume;

            if (inputLength != outputLength)
            {
                throw new ArgumentException("Input and output shapes have different lengths.");
            }

            return _outputShape;
        }
    }
}
