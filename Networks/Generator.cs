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
        private FeatureMap[,] FirstInGradients { get; set; }
        private FeatureMap[,] Outputs { get; set; }

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
                    string dataToLoad = "";
                    using (FileStream stream = new(file, FileMode.Open))
                    {
                        using (StreamReader read = new(stream))
                        {
                            dataToLoad = read.ReadToEnd();
                        }
                    }
                    generator = JsonConvert.DeserializeObject<Generator>(dataToLoad, new JsonSerializerSettings
                    {
                        TypeNameHandling = TypeNameHandling.Auto
                    });
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
        /// <param name="input">The images and their associatd labels for this iteration of training.</param>
        /// <param name="learningRate">The learning rate defining the degree to which each layer should be updated.</param>
        public void Backwards(FeatureMap[,] gradients, ImageInput[] input, float learningRate)
        {
            FeatureMap[,] images = new FeatureMap[1, _batchSize];
            for (int i = 0; i < _batchSize; i++)
            {
                images[0, i] = input[i].Image;
            }

            for (int i = 0; i < _batchSize; i++)
            {
                FirstInGradients[0, i] = gradients[0, i];
            }

            _updateStep++;
            float correctionLearningRate = CorrectionLearningRate(learningRate, 0.9f, 0.99f);

            for (int i = Depth - 1; i > 0; i--)
            {
                Utility.StopWatch(() => _layers[i].Backwards(correctionLearningRate, 0.9f, 0.999f), $"Backwards {i} {_layers[i].Name}", PRINTSTOPWATCH);
            }

            if (_layers[0] is not Convolution convolution)
            {
                Utility.StopWatch(() => _layers[0].Backwards(correctionLearningRate, 0.9f, 0.999f), $"Backwards {0} {_layers[0].Name}", PRINTSTOPWATCH);
            }
            else
            {
                Utility.StopWatch(() => convolution.BackwardsUpdateOnly(correctionLearningRate, 0.9f, 0.999f), $"Backwards {0} {_layers[0].Name}", PRINTSTOPWATCH);
            }
        }

        /// <summary>
        /// Forward propagates through the network to generate a new <see cref="FeatureMap"/> image from the starting image.
        /// </summary>
        /// <param name="input">The images and their asscoiated labels.</param>
        /// <param name="inference">Determines whether the <see cref="Generator"/> is training or inferring. Defaults to false.</param>
        public FeatureMap[,] Forward(ImageInput[] input, bool inference = false)
        {
            for (int i = 0; i < _batchSize; i++)
            {
                _inputImages[0, i] = input[i].Image;
                _classificationBools[i] = input[i].Bools;
                _classificationFloats[i] = input[i].Floats;
            }

            for (int i = 0; i < Depth; i++)
            {
                if (inference && _layers[i] is Dropout)
                {
                    Utility.StopWatch(() => (_layers[i] as Dropout).ForwardInference(), $"Forwards {i} {_layers[i].Name}", PRINTSTOPWATCH);
                }
                else
                {
                    Utility.StopWatch(() => _layers[i].Forward(), $"Forwards {i} {_layers[i].Name}", PRINTSTOPWATCH);
                }
            }

            return Outputs;
        }

        /// <inheritdoc/>
        public override void StartUp(int batchSize, int width, int length, int boolsLength, int floatsLength)
        {
            base.StartUp(batchSize, width, length, boolsLength, floatsLength);

            //_layers.Insert(0, new DropoutLayer(0.5f));
            _layers.Remove(_layers.FindLast(x => x is ReLUActivation));
            _layers.Remove(_layers.FindLast(x => x is BatchNormalization));

            _classificationBools = new bool[_batchSize][];
            _classificationFloats = new float[_batchSize][];
            for (int i = 0; i < batchSize; i++)
            {
                _inputImages[0, i] = new FeatureMap(width, length);
                _classificationBools[i] = new bool[boolsLength];
                _classificationFloats[i] = new float[floatsLength];
            }

            FeatureMap[,] current = _inputImages;
            FeatureMap[,] gradients = new FeatureMap[1, batchSize];

            foreach (var layer in _layers)
            {
                if (layer is LatentConvolution key)
                {
                    key.Bools = _classificationBools;
                    key.Floats = _classificationFloats;
                }

                (current, gradients) = layer.Startup(current, gradients);
            }

            Outputs = current;
            FirstInGradients = gradients;

            _ready = true;
        }
    }
}