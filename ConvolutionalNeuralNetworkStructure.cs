using Newtonsoft.Json;
using System.Runtime.Serialization;

public abstract partial class ConvolutionalNeuralNetwork
{
    protected FeatureMap[,] _inputImages;
    [JsonProperty] protected readonly List<ILayer> _layers = new();

    private readonly List<IPrimaryLayer> _primaryLayers = new();

    [JsonProperty] private List<(int, int)> _skipConnections;

    private ActivationPattern _activationPattern;
    protected int _batchSize;
    protected int _classificationBoolsLength;
    protected int _classificationFloatsLength;

    [JsonProperty] protected bool _configured = false;
    protected bool _ready = false;

    [JsonIgnore] public IEnumerable<IPrimaryLayer> PrimaryLayers
    {
        get
        {
            foreach (var layer in _primaryLayers)
                yield return layer;
        }
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

    public virtual void ResetNetwork()
    {
        foreach(var layer in _layers)
        {
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

    [OnSerializing]
    private void OnSerializing(StreamingContext context) {
        _skipConnections = new List<(int, int)>();
        for(int i = 0; i < _layers.Count; i++)
        {
            if (_layers[i] is SkipConnectionLayer skip)
            {
                _skipConnections.Add((i, _layers.IndexOf(skip.GetConcatenationLayer())));
            }
        }

    }

    [OnDeserialized]
    private void OnDeserialized(StreamingContext context)
    {
        if (_skipConnections != null)
        {
            foreach((int skipIndex, int concatIndex) in _skipConnections)
            {
                SkipConnectionLayer skip = new SkipConnectionLayer();
                ConcatenationLayer concat = skip.GetConcatenationLayer();
                _layers[skipIndex] = skip;
                _layers[concatIndex] = concat;
            }
        }
    }

    public virtual void StartUp(int batchSize, int width, int length, int boolsLength, int floatsLength)
    {
        if (!_configured)
        {
            if (_activationPattern.Equals(default(ActivationPattern)))
            {
                _activationPattern = new ActivationPattern(new NormalizationLayers[] {
                    NormalizationLayers.Activation,
                    NormalizationLayers.BatchNormalization
                }, 0);
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
        _classificationBoolsLength = boolsLength;
        _classificationFloatsLength = floatsLength;
    }
}