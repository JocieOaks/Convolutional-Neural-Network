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

        public override void Backwards(int batchSize)
        {
        }

        public override void Forward(int batchSize)
        {
        }

        public override void Reset()
        {
        }

        public override Shape Startup(Shape inputShape, IOBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            int inputLength = inputShape.Volume;

            int outputLength = _outputShape.Volume;

            if(inputLength != outputLength)
            {
                throw new ArgumentException("Input and output shapes have different lengths.");
            }

            return _outputShape;
        }
    }
}
