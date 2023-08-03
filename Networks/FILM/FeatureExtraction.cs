using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ConvolutionalNeuralNetwork.Layers;
using ConvolutionalNeuralNetwork.Layers.Initializers;
using ConvolutionalNeuralNetwork.Layers.Skip;
using ILGPU.Runtime;
using Newtonsoft.Json;
using ConvolutionalNeuralNetwork.Layers.Activations;
using ConvolutionalNeuralNetwork.Layers.Weighted;

namespace ConvolutionalNeuralNetwork.Networks
{
    public partial class FILM
    {
        [Serializable]
        private class FeatureExtraction : Network
        {
            [JsonProperty] private readonly List<Layer>[] _featureLayers;
            private readonly SkipSplit[] _outputLayers;
            [JsonIgnore] public SkipSplit[] OutputLayers => _outputLayers;

            private int _inputArea;

            public FeatureExtraction(ConvolutionSharedWeights[] shared)
            {
                _featureLayers = new List<Layer>[PYRAMIDLAYERS];
                _outputLayers = new SkipSplit[PYRAMIDLAYERS];

                List<SkipSplit>[] skips = new List<SkipSplit>[PYRAMIDLAYERS];
                for (int i = 0; i < PYRAMIDLAYERS; i++)
                {
                    skips[i] = new();
                }

                for (int i = 0; i < PYRAMIDLAYERS; i++)
                {
                    _featureLayers[i] = new List<Layer>();
                    if (i != 0)
                    {
                        _featureLayers[i].Add(new AveragePool((int)Math.Pow(2, i)));
                    }
                    for (int j = 0; j < 3; j++)
                    {
                        if (i + j < PYRAMIDLAYERS)
                        {
                            _featureLayers[i].Add(new Convolution(shared[j], null));
                            _featureLayers[i].Add(new ReLUActivation());
                            _featureLayers[i].Add(new BatchNormalization());
                            _featureLayers[i].Add(new Convolution(shared[j + 3], null));
                            _featureLayers[i].Add(new ReLUActivation());
                            _featureLayers[i].Add(new BatchNormalization());
                            var skip = new SkipSplit();
                            skips[i + j].Add(skip);
                            _featureLayers[i].Add(skip);
                            _featureLayers[i].Add(new AveragePool(2));
                        }
                    }
                }

                for(int i = 0; i < PYRAMIDLAYERS; i++)
                {
                    _layers.Add(skips[i].First().GetOutLayer());
                    foreach(var skip in skips[i].Skip(1))
                    {
                        _layers.Add(skip.GetConcatenationLayer());
                    }
                    var output = new SkipSplit();
                    _outputLayers[i] = output;
                    _layers.Add(output);
                }

            }

            [JsonConstructor] private FeatureExtraction() { }

            public override void StartUp(int maxBatchSize, int inputWidth, int inputLength, int boolLabels, int floatLabels, AdamHyperParameters hyperParameters, int inputChannels)
            {
                base.StartUp(maxBatchSize, inputWidth, inputLength, boolLabels, floatLabels, hyperParameters, inputChannels);

                _inputArea = inputWidth * inputLength;

                _startBuffers ??= new();
                _middleBuffers ??= new();
                _middleBuffers.OutputDimensionArea(inputWidth * inputLength * inputChannels);
                Shape current;
                IOBuffers inputBuffers;
                IOBuffers outputBuffers;

                for (int i = 0; i < PYRAMIDLAYERS; i++)
                {
                    current = new(inputWidth, inputLength, inputChannels);
                    inputBuffers = _startBuffers;
                    outputBuffers = _middleBuffers;
                    foreach (var layer in _featureLayers[i])
                    {
                        current = layer.Startup(current, inputBuffers, maxBatchSize);
                        if (layer is WeightedLayer weighted)
                        {
                            foreach(var weight in weighted.SetUpWeights())
                            _weights.Add(weight);
                        }
                        else if (layer is BatchNormalization bn)
                        {
                            bn.SetHyperParameters(_adamHyperParameters);
                        }
                        if (layer is not IUnchangedLayer)
                        {
                            (inputBuffers, outputBuffers) = (outputBuffers, inputBuffers);
                        }
                    }
                }

                current = new Shape();
                inputBuffers = _startBuffers;
                outputBuffers = _middleBuffers;
                foreach(var layer in _layers)
                {
                    current = layer.Startup(current, inputBuffers, maxBatchSize);
                    if (layer is not IUnchangedLayer)
                    {
                        (inputBuffers, outputBuffers) = (outputBuffers, inputBuffers);
                    }
                }
            }

            /// <summary>
            /// Loads a <see cref="FeatureExtraction"/> from a json file.
            /// </summary>
            /// <param name="file">The path of the json file.</param>
            /// <returns>Returns the deserialized <see cref="FeatureExtraction"/>.</returns>
            public static FeatureExtraction LoadFromFile(string file)
            {
                FeatureExtraction feature = null;

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
                                feature = serializer.Deserialize<FeatureExtraction>(reader);
                            }
                        }
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine("Error occured when trying to load data from file: " + file + "\n" + e.ToString());
                    }
                }

                return feature;
            }

            public List<Weights> GetWeights()
            {
                return _weights;
            }

            public void Forward(FeatureMap[][] images)
            {
                int batchSize = images.Length;
                for (int i = 0; i < PYRAMIDLAYERS; i++)
                {
                    for (int j = 0; j < batchSize; j++)
                    {
                        for (int k = 0; k < _inputChannels; k++)
                        {
                            if (images[j][k].Area != _inputArea)
                            {
                                throw new ArgumentException("Input images are incorrectly sized.");
                            }
                            images[j][k].CopyToBuffer(_startBuffers.Input.SubView(_inputArea * (j * _inputChannels + k), _inputArea));
                        }
                    }

                    for (int j = 0; j < _featureLayers[i].Count; j++)
                    {
                        Utility.StopWatch(() => _featureLayers[i][j].Forward(batchSize), $"Forwards {i} {j} {_featureLayers[i][j].Name}", PRINTSTOPWATCH);
                    }
                }

                for (int j = 0; j < Depth; j++)
                {
                    Utility.StopWatch(() => _layers[j].Forward(batchSize), $"Forwards C {j} {_layers[j].Name}", PRINTSTOPWATCH);
                }
            }

            public void Backwards(int batchSize)
            {
                for (int i = PYRAMIDLAYERS - 1; i >= 0; i--)
                {
                    _featureLayers[i].Last().InGradient.MemSetToZero();
                    for (int j = _featureLayers[i].Count - 1; j >= 0; j--)
                    {
                        Utility.StopWatch(() => _featureLayers[i][j].Backwards(batchSize, true), $"Backwards {i} {j} {_featureLayers[i][j].Name}", PRINTSTOPWATCH);
                    }
                }

                for (int j = Depth - 1; j >= 0; j--)
                {
                    Utility.StopWatch(() => _layers[j].Backwards(batchSize, true), $"Backwards C {j} {_layers[j].Name}", PRINTSTOPWATCH);
                }
            }
        }
    }
}
