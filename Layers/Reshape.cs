using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers
{
    public class Reshape : Layer, IStructuralLayer, IUnchangedLayer
    {
        [JsonProperty] public Shape[] OutputShapes { get => _outputShapes; set => _outputShapes = value; }

        public Reshape(Shape[] outputShapes)
        {
            _outputShapes = outputShapes;
        }

        [JsonConstructor] private Reshape() { }

        public override string Name => "Reshape Layer";

        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
        }

        public override void Forward()
        {
        }

        public override void Reset()
        {
        }

        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, int batchSize)
        {

            _inputDimensions = inputShapes.Length;
            _outputDimensions = _outputShapes.Length;

            int inputLength = 0;
            for(int i = 0; i < _inputDimensions; i++)
            {
                inputLength += inputShapes[i].Area;
            }

            int outputLength = 0;
            for(int i = 0; i < _outputDimensions; i++)
            {
                outputLength += _outputShapes[i].Area;
            }

            if(inputLength != outputLength)
            {
                throw new ArgumentException("Input and output shapes have different lengths.");
            }

            return _outputShapes;
        }
    }
}
