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
    partial class FILM
    {
        private class Fusion : Network
        {
            public Fusion(SkipSplit[] f0, SkipSplit[] f1, SkipSplit[] flow0, SkipSplit[] flow1)
            {
                _layers.Add(f0[^1].GetOutLayer());
                _layers.Add(f1[^1].GetConcatenationLayer());
                _layers.Add(flow0[^1].GetConcatenationLayer());
                _layers.Add(flow1[^1].GetConcatenationLayer());
                _layers.Add(new TransposeConvolution(4, 2, 16, GlorotUniform.Instance));

                for(int i = PYRAMIDLAYERS - 2; i >= 0; i--)
                {
                    _layers.Add(f0[i].GetConcatenationLayer());
                    _layers.Add(f1[i].GetConcatenationLayer());
                    _layers.Add(flow0[i].GetConcatenationLayer());
                    _layers.Add(flow1[i].GetConcatenationLayer());
                    _layers.Add(new Convolution(3, 1, 16, GlorotUniform.Instance));
                    _layers.Add(new ReLUActivation());
                    _layers.Add(new Convolution(3, 1, 16, GlorotUniform.Instance));
                    _layers.Add(new ReLUActivation());
                    _layers.Add(new TransposeConvolution(4, 2, 16, GlorotUniform.Instance));
                    _layers.Add(new ReLUActivation());
                }

                _layers.Add(new Convolution(1, 1, 4, GlorotUniform.Instance, false));
                _layers.Add(new HyperTan());
            }

            [JsonConstructor] private Fusion() { }

            /// <summary>
            /// Loads a <see cref="Fusion"/> from a json file.
            /// </summary>
            /// <param name="file">The path of the json file.</param>
            /// <returns>Returns the deserialized <see cref="Fusion"/>.</returns>
            public static Fusion LoadFromFile(string file)
            {
                Fusion fusion = null;

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
                                fusion = serializer.Deserialize<Fusion>(reader);
                            }
                        }
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine("Error occured when trying to load data from file: " + file + "\n" + e.ToString());
                    }
                }

                return fusion;
            }

            public override void StartUp(int maxBatchSize, int inputWidth, int inputLength, int boolLabels, int floatLabels, AdamHyperParameters hyperParameters, int inputChannels)
            {
                base.StartUp(maxBatchSize, inputWidth, inputLength, boolLabels, floatLabels, hyperParameters, inputChannels);

                Shape shape = new();

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
        }
    }
}
