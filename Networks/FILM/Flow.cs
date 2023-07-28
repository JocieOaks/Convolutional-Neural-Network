using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers;
using ConvolutionalNeuralNetwork.Layers.Initializers;
using ConvolutionalNeuralNetwork.Layers.Skip;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ConvolutionalNeuralNetwork.Layers.Activations;
using ConvolutionalNeuralNetwork.Layers.Weighted;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Networks
{
    public partial class FILM
    {
        private class Flow : Network
        {

            public Flow(SkipSplit[] f1, SkipSplit[] f0)
            {
                _layers.Add(f1[^1].GetOutLayer());
                _layers.Add(f0[^1].GetConcatenationLayer());
                _layers.Add(new Convolution(3, 1, (int)Math.Pow(2, PYRAMIDLAYERS + 4), GlorotUniform.Instance));
                _layers.Add(new ReLUActivation());
                _layers.Add(new Convolution(3, 1, (int)Math.Pow(2, PYRAMIDLAYERS + 3), GlorotUniform.Instance));
                _layers.Add(new ReLUActivation());
                _layers.Add(new Convolution(3, 1, 2, GlorotUniform.Instance));
                _layers.Add(new ReLUActivation());
                var skip = new SkipSplit();
                _layers.Add(skip);
                _flow[^1] = skip;
                var transposeInit = new Constant(1);
                _layers.Add(new TransposeConvolution(2, 2, 2, transposeInit, false));
                skip = new SkipSplit();
                _layers.Add(skip);
                _flowUpsampled[^1] = skip;

                for (int i = PYRAMIDLAYERS - 2; i >= 0; i--)
                {
                    _layers.Add(f1[i].GetConcatenationLayer());
                    _layers.Add(new Warp());
                    _layers.Add(f0[i].GetConcatenationLayer());
                    _layers.Add(new Convolution(3, 1, (int)Math.Pow(2, i + 4), GlorotUniform.Instance));
                    _layers.Add(new ReLUActivation());
                    _layers.Add(new Convolution(3, 1, (int)Math.Pow(2, i + 3), GlorotUniform.Instance));
                    _layers.Add(new ReLUActivation());
                    _layers.Add(new Convolution(3, 1, 2, GlorotUniform.Instance));
                    _layers.Add(new ReLUActivation());
                    _layers.Add(_flowUpsampled[i].GetConcatenationLayer());
                    var sum = new Summation();
                    sum.SetOutputDimensions(2);
                    _layers.Add(sum);
                    skip = new SkipSplit();
                    _layers.Add(skip);
                    _flow[i] = skip;
                    if (i != 0)
                    {
                        _layers.Add(new TransposeConvolution(2, 2, 2, transposeInit, false));
                        skip = new SkipSplit();
                        _layers.Add(skip);
                        _flowUpsampled[i - 1] = skip;
                    }
                }

                for (int i = 0; i < PYRAMIDLAYERS; i++)
                {
                    _layers.Add(_flow[i].GetOutLayer());
                    _layers.Add(f1[i].GetConcatenationLayer());
                    _layers.Add(new Warp());
                    skip = new SkipSplit();
                    _layers.Add(skip);
                    _f[i] = skip;
                }
            }

            [JsonConstructor] private Flow() { }

            private readonly SkipSplit[] _flow = new SkipSplit[PYRAMIDLAYERS];
            private readonly SkipSplit[] _flowUpsampled = new SkipSplit[PYRAMIDLAYERS - 1];

            public SkipSplit[] Flows => _flow;

            private readonly SkipSplit[] _f = new SkipSplit[PYRAMIDLAYERS];

            public SkipSplit[] F => _f;

            public override void StartUp(int maxBatchSize, int inputWidth, int inputLength, int boolLabels, int floatLabels, AdamHyperParameters hyperParameters, int inputChannels)
            {
                base.StartUp(maxBatchSize, inputWidth, inputLength, boolLabels, floatLabels, hyperParameters, inputChannels);

                Shape shape = new Shape();
                InitializeLayers(ref shape, maxBatchSize);
            }

            public void Forward(int batchSize)
            {
                for (int j = 0; j < Depth; j++)
                {
                    Utility.StopWatch(() => _layers[j].Forward(batchSize), $"Forwards {j} {_layers[j].Name}", PRINTSTOPWATCH);
                }
            }

            public void Backwards(int batchSize)
            {
                _adamHyperParameters.Update();

                for (int j = Depth - 1; j >= 0; j--)
                {
                    Utility.StopWatch(() => _layers[j].Backwards(batchSize), $"Backwards {j} {_layers[j].Name}", PRINTSTOPWATCH);
                }
            }

            /// <summary>
            /// Loads a <see cref="Flow"/> from a json file.
            /// </summary>
            /// <param name="file">The path of the json file.</param>
            /// <returns>Returns the deserialized <see cref="Flow"/>.</returns>
            public static Flow LoadFromFile(string file)
            {
                Flow flow = null;

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
                                flow = serializer.Deserialize<Flow>(reader);
                            }
                        }
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine("Error occured when trying to load data from file: " + file + "\n" + e.ToString());
                    }
                }

                return flow;
            }
        }
    }
}
