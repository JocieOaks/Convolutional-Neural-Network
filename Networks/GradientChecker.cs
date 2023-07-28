using ConvolutionalNeuralNetwork.DataTypes;
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

        public override void StartUp(int maxBatchSize, int width, int length, int boolLabels, int floatLabels, AdamHyperParameters hyperParameters, int inputChannels)
        {
            base.StartUp(maxBatchSize, width, length, boolLabels, floatLabels, hyperParameters, 3);

            int inputArea = width * length;

            /*Shape[] current = new Shape[1];
            current[0] = new Shape(width, length);
            _finalOutGradient = new FeatureMap[batchSize];
            _finalOutput = new Vector[batchSize];
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

            _discriminatorGradients = new Vector[batchSize];
            _generatorGradients = new Vector[batchSize];

            _ready = true;*/
        }
    }
}
