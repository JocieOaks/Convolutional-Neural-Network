using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Networks
{
    /// <summary>
    /// The <see cref="Generator"/> is a <see cref="Network"/> for generating images from a starting image.
    /// </summary>
    public class Generator : Network
    {
        private bool[][] _classificationBools;
        private float[][] _classificationFloats;
        private FeatureMap[] _outputs;
        private FeatureMap[,] _latentSpace;
        private readonly int _latentDimensions = 50;
        private int _outputArea;

        /// <summary>
        /// Loads a <see cref="Generator"/> from a json file.
        /// </summary>
        /// <param name="file">The path of the json file.</param>
        /// <returns>Returns the deserialized <see cref="Generator"/>.</returns>
        public static Generator LoadFromFile(string file)
        {
            Generator generator = null;

            if (File.Exists(file))
            {
                try
                {
                    using (StreamReader r = new(file))
                    {
                        using (JsonReader reader = new JsonTextReader(r))
                        {
                            JsonSerializer serializer = new();
                            serializer.TypeNameHandling = TypeNameHandling.Auto;
                            generator = serializer.Deserialize<Generator>(reader);
                        }
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine("Error occured when trying to load data from file: " + file + "\n" + e.ToString());
                }
            }

            return generator;
        }

        /// <summary>
        /// Backpropogates through the network, updating every <see cref="ILayer"/> in the <see cref="Generator"/>.
        /// </summary>
        /// <param name="gradients">An array of <see cref="FeatureMap"/>'s containing the gradients for the last layer of the <see cref="Generator"/>.</param>
        /// <param name="inputs">The images and their associatd labels for this iteration of training.</param>
        /// <param name="learningRate">The learning rate defining the degree to which each layer should be updated.</param>
        public void Backwards(FeatureMap[] gradients)
        {
            HyperTan.Backward(_outputs, gradients);
            for (int i = 0; i < _batchSize; i++)
            {
                gradients[i].CopyToBuffer(_endBuffers.InGradient.SubView(_outputArea * i, _outputArea));
            }

            _updateStep++;
            float correctionLearningRate = CorrectionLearningRate(_learningRate, _firstMomentDecay, _secondMomentDecay);

            for (int i = Depth - 1; i >= 0; i--)
            {
                Utility.StopWatch(() => _layers[i].Backwards(correctionLearningRate, _firstMomentDecay, _secondMomentDecay), $"Backwards {i} {_layers[i].Name}", true);
            }
        }

        /// <summary>
        /// Forward propagates through the network to generate a new <see cref="FeatureMap"/> image from the starting image.
        /// </summary>
        /// <param name="input">The images and their asscoiated labels.</param>
        /// <param name="inference">Determines whether the <see cref="Generator"/> is training or inferring. Defaults to false.</param>
        public FeatureMap[] Forward(ImageInput[] input)
        {

            for (int i = 0; i < _batchSize; i++)
            {
                _classificationBools[i] = input[i].Bools;
                _classificationFloats[i] = input[i].Floats;
            }

            for(int i = 0; i < _latentDimensions; i++)
            {
                for(int j = 0; j < _batchSize; j++)
                {
                    _latentSpace[i, j].Randomize(0, 1);
                    _latentSpace[i, j].CopyToBuffer(_startBuffers.Input.SubView(j * _latentDimensions + i, 1));
                }
            }

            for (int i = 0; i < Depth; i++)
            {
                Utility.StopWatch(() => _layers[i].Forward(), $"Forwards {i} {_layers[i].Name}", true);
            }

            for(int i = 0; i < _batchSize; i++)
            {
                _outputs[i].SyncCPU(_endBuffers.Output.SubView(_outputArea * i, _outputArea));
            }

            HyperTan.Forward(_outputs);

            return _outputs;
        }

        /// <inheritdoc/>
        public override void StartUp(int batchSize, int width, int length, int labelBools, int labelFloats, float learningRate, float firstDecay, float secondDecay)
        {
            base.StartUp(batchSize, width, length, labelBools, labelFloats, learningRate, firstDecay, secondDecay);

            for(int i = _layers.Count - 1; i >= 0; i--)
            {
                if (_layers[i] is IPrimaryLayer && _layers[i] is not IStructuralLayer)
                {
                    break;
                }
                if (_layers[i] is ReLUActivation || _layers[i] is BatchNormalization)
                    _layers.Remove(_layers[i]);
            }

            _outputs = new FeatureMap[batchSize];
            _outputArea = width * length;

            _classificationBools = new bool[_batchSize][];
            _classificationFloats = new float[_batchSize][];
            for (int i = 0; i < batchSize; i++)
            {
                _outputs[i] = new FeatureMap(width, length);
                _classificationBools[i] = new bool[labelBools];
                _classificationFloats[i] = new float[labelFloats];
            }

            _latentSpace = new FeatureMap[_latentDimensions, _batchSize];
            for(int i = 0; i < _latentDimensions; i++)
            {
                for(int j = 0; j < _batchSize; j++)
                {
                    _latentSpace[i, j] = new FeatureMap(1, 1);
                }
            }

            Shape current  = new Shape(1, 1, _latentDimensions);
            IOBuffers inputBuffers = _startBuffers = new();
            IOBuffers outputBuffers = new();
            outputBuffers.OutputDimensionArea(_latentDimensions);

            foreach (var layer in _layers)
            {
                /*if (layer is LatentConvolution key)
                {
                    key.Bools = _classificationBools;
                    key.Floats = _classificationFloats;
                }*/

                current = layer.Startup(current, inputBuffers, batchSize);
                if (layer is not IUnchangedLayer)
                {
                    (inputBuffers, outputBuffers) = (outputBuffers, inputBuffers); 
                }
            }
            _endBuffers = outputBuffers;
            _middleBuffers = inputBuffers;
            inputBuffers.Allocate(batchSize);
            outputBuffers.Allocate(batchSize);
            IOBuffers.SetCompliment(inputBuffers, outputBuffers);

            _ready = true;
        }
    }
}