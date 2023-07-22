using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Networks
{
    public class GradientChecker : Network
    {
        FeatureMap[,] _input;
        FeatureMap[,] _output;

        public override void StartUp(int batchSize, int width, int length, int boolLabels, int floatLabels, float learningRate, float firstDecay, float secondDecay)
        {
            base.StartUp(batchSize, width, length, boolLabels, floatLabels, learningRate, firstDecay, secondDecay);

            int inputArea = width * length;

            /*Shape[] current = new Shape[1];
            current[0] = new Shape(width, length);
            _finalOutGradient = new FeatureMap[batchSize];
            _finalOutput = new Vector[_batchSize];
            for (int j = 0; j < batchSize; j++)
            {
                _finalOutGradient[j] = new FeatureMap(width, length);
                _finalOutput[j] = new Vector(labelBools + labelFloats);
            }

            _startBuffers ??= new();
            _middleBuffers ??= new();

            IOBuffers inputBuffers = _startBuffers;
            IOBuffers outputBuffers = _middleBuffers;
            outputBuffers.OutputDimensionArea(width * length);

            foreach (var layer in _layers)
            {
                current = layer.Startup(current, inputBuffers, batchSize);
                if (layer is not IUnchangedLayer)
                {
                    (inputBuffers, outputBuffers) = (outputBuffers, inputBuffers);
                }
            }

            _endBuffers = outputBuffers;

            inputBuffers.Allocate(batchSize);
            outputBuffers.Allocate(batchSize);
            IOBuffers.SetCompliment(inputBuffers, outputBuffers);

            _discriminatorGradients = new Vector[_batchSize];
            _generatorGradients = new Vector[_batchSize];

            _ready = true;*/
        }
    }
}
