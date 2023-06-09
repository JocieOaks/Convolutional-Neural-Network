using Newtonsoft.Json;

public partial class ConvolutionalNeuralNetwork
{
    private FeatureMap[,] _inputImages;
    private FeatureMap[,] _finalOutGradient;
    [JsonProperty] private readonly List<ILayer> _layers = new();

    private readonly List<IPrimaryLayer> _primaryLayers = new();
    [JsonProperty] private readonly Transformer _transformer;

    [JsonProperty] private readonly VectorizationLayer _vectorizationLayer;

    private ActivationPattern _activationPattern;
    private int _batchSize;

    [JsonProperty] private bool _configured = false;
    private bool _ready = false;

    public ConvolutionalNeuralNetwork(int vectorDimensions)
    {
        _transformer = new Transformer(vectorDimensions);
        _vectorizationLayer = new VectorizationLayer(vectorDimensions);
    }

    public IEnumerable<IPrimaryLayer> PrimaryLayers
    {
        get
        {
            foreach (var layer in _primaryLayers)
                yield return layer;
        }
    }

    public static ConvolutionalNeuralNetwork LoadFromFile(string file)
    {
        ConvolutionalNeuralNetwork cnn = null;

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
                cnn = JsonConvert.DeserializeObject<ConvolutionalNeuralNetwork>(dataToLoad, new JsonSerializerSettings
                {
                    TypeNameHandling = TypeNameHandling.Auto
                });
            }
            catch (Exception e)
            {
                Console.WriteLine("Error occured when trying to load data from file: " + file + "\n" + e.ToString());
            }
        }

        return cnn;
    }

    public void AddLayer(IPrimaryLayer layer)
    {
        _primaryLayers.Add(layer);
    }

    public void AddSkipConnection(int index1, int index2)
    {
        SkipConnectionLayer skipLayer = new SkipConnectionLayer();
        ConcatenationLayer concatenationLayer = skipLayer.GetConcatenationLayer();
        _primaryLayers.Insert(index2, concatenationLayer);
        _primaryLayers.Insert(index1, skipLayer);
    }

    public void ChangeVectorDimensions(int dimensions)
    {
        _vectorizationLayer.ChangeVectorDimensions(dimensions);
        _transformer.ChangeVectorDimensions(dimensions);
    }

    public void ClearLayers()
    {
        _primaryLayers.Clear();
        _layers.Clear();
    }

    public void DeleteLayer(IPrimaryLayer layer)
    {
        _primaryLayers.Remove(layer);
    }

    public void DeleteLayer(int index)
    {
        _primaryLayers.RemoveAt(index);
    }

    public void DeleteLayers(Predicate<IPrimaryLayer> predicate)
    {
        _primaryLayers.RemoveAll(predicate);
    }

    public void InsertLayer(IPrimaryLayer layer, int index)
    {
        _primaryLayers.Insert(index, layer);
    }

    public void ReconfigureNetwork()
    {
        _configured = false;
        foreach (var layer in _layers)
        {
            if (layer is IPrimaryLayer primary)
                _primaryLayers.Add(primary);
        }
        _layers.Clear();
    }

    public void ResetLayer(int index)
    {
        _layers[index].Reset();
    }

    public void ResetAll(Predicate<ILayer> predicate)
    {
        foreach (var layer in _layers)
        {
            if (predicate(layer))
                layer.Reset();
        }
    }

    public void SaveToFile(string file)
    {
        try
        {
            // create the directory the file will be written to if it doesn't already exist
            Directory.CreateDirectory(Path.GetDirectoryName(file)!);

            // serialize the C# game data object into Json
            string dataToStore = JsonConvert.SerializeObject(this, Formatting.Indented, new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.Auto
            });

            // write the serialized data to the file
            using (FileStream stream = File.Create(file))
            {
                using (StreamWriter writer = new(stream))
                {
                    writer.Write(dataToStore);
                }
            }
        }
        catch (System.Exception e)
        {
            Console.WriteLine("Error occured when trying to save data to file: " + file + "\n" + e.ToString());
        }
    }

    public void SetActivationPattern(ActivationPattern pattern)
    {
        _activationPattern = pattern;
    }

    public bool StartUp(int batchSize, int width, int length, int descriptionBools, int descriptionFloats)
    {
        if (!_configured)
        {
            if (_activationPattern.Equals(default(ActivationPattern)))
            {
                _activationPattern = new ActivationPattern(new NormalizationLayers[] {
                    NormalizationLayers.Activation,
                    NormalizationLayers.Dropout,
                    NormalizationLayers.BatchNormalization
                }, 0.2f);
            }

            foreach (var primaryLayer in _primaryLayers)
            {
                _layers.Add(primaryLayer);
                if (primaryLayer is not IStructuralLayer)
                {
                    foreach (var secondaryLayer in _activationPattern.GetLayers())
                    {
                        _layers.Add(secondaryLayer);
                    }
                }
            }

            _configured = true;
        }

        _batchSize = batchSize;
        _inputImages = new FeatureMap[1, batchSize];
        for (int j = 0; j < batchSize; j++)
        {
            _inputImages[0, j] = new FeatureMap(width, length);
        }

        FeatureMap[,] current = _inputImages;
        FeatureMap[,] gradients = new FeatureMap[1, batchSize];
        foreach (var layer in _layers)
        {
            (current, gradients) = layer.Startup(current, gradients);
        }

        _vectorizationLayer.StartUp(TransposeArray(current), gradients);

        _descriptionVectors = new Vector[batchSize];
        _descriptionVectorsNorm = new Vector[batchSize];

        _ready = true;
        return _transformer.Startup(descriptionBools, descriptionFloats);
    }
}