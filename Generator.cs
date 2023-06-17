using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class Generator : ConvolutionalNeuralNetwork
{
    private FeatureMap[,] FirstInGradients { get; set; }

    private bool[][] _classificationBools;
    private float[][] _classificationFloats;

    private FeatureMap[,] Outputs { get; set; }

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

        for (int i = Depth - 1; i >= 0; i--)
        {
            StopWatch(() => _layers[i].Backwards(learningRate), $"Backwards {i} {_layers[i].Name}");
        }
    }

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
            if (inference && _layers[i] is DropoutLayer)
            {
                StopWatch(() => (_layers[i] as DropoutLayer).ForwardInference(), $"Forwards {i} {_layers[i].Name}");
            }
            else
            {
                StopWatch(() => _layers[i].Forward(), $"Forwards {i} {_layers[i].Name}");
            }
        }

        return Outputs;
    }

    public override void StartUp(int batchSize, int width, int length, int boolsLength, int floatsLength)
    {
        base.StartUp(batchSize, width, length, boolsLength, floatsLength);

        //_layers.Insert(0, new DropoutLayer(0.5f));
        _layers.Remove(_layers.FindLast(x => x is ReLULayer));
        _layers.Remove(_layers.FindLast(x => x is BatchNormalizationLayer));

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
            if(layer is ConvolutionalKeyLayer key)
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
}

