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
        [JsonProperty] public Shape OutputShape { get => _outputShape; set => _outputShape = value; }

        public Reshape(Shape outputShape)
        {
            _outputShape = outputShape;
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

        public override Shape Startup(Shape inputShapes, IOBuffers buffers, int batchSize)
        {

            _inputDimensions = inputShapes.Dimensions;
            _outputDimensions = _outputShape.Dimensions;

            int inputLength = inputShapes.Area * _inputDimensions;

            int outputLength = _outputShape.Area * _outputDimensions;

            if(inputLength != outputLength)
            {
                throw new ArgumentException("Input and output shapes have different lengths.");
            }

            return _outputShape;
        }
    }
}
