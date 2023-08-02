using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Design;
using ConvolutionalNeuralNetwork.Layers;
using Newtonsoft.Json;
using System.Runtime.Serialization;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ConvolutionalNeuralNetwork.Layers.Skip;
using ConvolutionalNeuralNetwork.Layers.Weighted;

namespace ConvolutionalNeuralNetwork
{
    /// <summary>
    /// The <see cref="Network"/> class is the base class for all Convolutional Neural Networks.
    /// </summary>
    public abstract class Network
    {
        protected const bool PRINTSTOPWATCH = false;

        [JsonProperty] protected readonly List<ILayer> _layers = new();
        protected int _boolLabels;
        [JsonProperty] protected bool _configured = false;
        protected int _floatLabels;
        protected bool _ready = false;
        private readonly List<IPrimaryLayer> _primaryLayers = new();
        [JsonProperty] protected readonly List<Weights> _weights = new();

        private ActivationPattern _activationPattern;

        protected IOBuffers _startBuffers;
        protected IOBuffers _middleBuffers;
        protected IOBuffers _endBuffers;
        [JsonProperty] protected AdamHyperParameters _adamHyperParameters;
        protected int _inputChannels;

        [JsonIgnore] public ArrayView<float> Input => _layers.First().Input;

        [JsonIgnore] public ArrayView<float> Output => _layers.Last().Output;

        [JsonIgnore] public ArrayView<float> InGradient => _layers.Last().InGradient;

        [JsonIgnore] public ArrayView<float> OutGradient => _layers.First().OutGradient;

        [JsonIgnore] protected int LabelCount => _floatLabels + _boolLabels;

        /// <value>Enumerates over the <see cref="IPrimaryLayer"/>'s of the <see cref="Network"/> that define its structure.</value>
        [JsonIgnore]
        public IEnumerable<IPrimaryLayer> PrimaryLayers
        {
            get
            {
                if (_primaryLayers == null)
                {
                    foreach (var layer in _layers)
                    {
                        if (layer is IPrimaryLayer primary)
                            _primaryLayers.Add(primary);
                    }
                }
                foreach (var layer in _primaryLayers)
                    yield return layer;
            }
        }

        /// <value>The number of <see cref="Layer"/>s in the <see cref="Network"/>.</value>
        protected int Depth => _layers.Count;

        /// <summary>
        /// Appends a new <see cref="IPrimaryLayer"/> to the end of the <see cref="Network"/>.
        /// </summary>
        /// <param name="layer">The <see cref="IPrimaryLayer"/> being added.</param>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="Network"/> is not being configured.</exception>
        public void AddLayer(IPrimaryLayer layer)
        {
            if (_configured)
                throw new InvalidOperationException("Network is already configured.");
            _primaryLayers.Add(layer);
        }

        /// <summary>
        /// Adds a <see cref="SkipSplit"/> and <see cref="SkipConcatenate"/> to connect two layers in the <see cref="Network"/>.
        /// </summary>
        /// <param name="index1">The index of the <see cref="SkipSplit"/>.</param>
        /// <param name="index2">The index of the <see cref="SkipConcatenate"/>.</param>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="Network"/> is not being configured.</exception>
        public void AddSkipConnection(int index1, int index2)
        {
            /*if (_configured)
                throw new InvalidOperationException("Network is already configured.");

            SkipConnectionSplit skipLayer = new();
            SkipConnectionConcatenate concatenationLayer = skipLayer.GetConcatenationLayer();
            _primaryLayers.Insert(index2, concatenationLayer);
            _primaryLayers.Insert(index1, skipLayer);*/
        }

        /// <summary>
        /// Removes all the <see cref="Layer"/>s from the <see cref="Network"/>.
        /// </summary>
        public void ClearLayers()
        {
            _primaryLayers.Clear();
            _layers.Clear();
        }

        /// <summary>
        /// Removes the specified <see cref="IPrimaryLayer"/> from the <see cref="Network"/>.
        /// </summary>
        /// <param name="layer">The <see cref="IPrimaryLayer"/> being removed.</param>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="Network"/> is not being configured.</exception>
        public void DeleteLayer(IPrimaryLayer layer)
        {
            if (_configured)
                throw new InvalidOperationException("Network is already configured.");

            _primaryLayers.Remove(layer);
        }

        /// <summary>
        /// Remove's the <see cref="IPrimaryLayer"/> at the specified index from the <see cref="Network"/>.
        /// </summary>
        /// <param name="index">The index of the <see cref="IPrimaryLayer"/> to be removed.</param>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="Network"/> is not being configured.</exception>
        public void DeleteLayer(int index)
        {
            if (_configured)
                throw new InvalidOperationException("Network is already configured.");

            _primaryLayers.RemoveAt(index);
        }

        /// <summary>
        /// Remove's all <see cref="IPrimaryLayer"/>s that match the conditions defined by the specified predicate
        /// from the <see cref="Network"/>.
        /// </summary>
        /// <param name="predicate">A predicate that dictate's which <see cref="IPrimaryLayer"/>s are removed.</param>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="Network"/> is not being configured.</exception>
        public void DeleteLayers(Predicate<IPrimaryLayer> predicate)
        {
            if (_configured)
                throw new InvalidOperationException("Network is already configured.");

            _primaryLayers.RemoveAll(predicate);
        }

        /// <summary>
        /// Adds a new <see cref="IPrimaryLayer"/> to the <see cref="Network"/> at the specified index.
        /// </summary>
        /// <param name="layer">The <see cref="IPrimaryLayer"/> to be added.</param>
        /// <param name="index">The index to insert the layer at.</param>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="Network"/> is not being configured.</exception>
        public void InsertLayer(IPrimaryLayer layer, int index)
        {
            if (_configured)
                throw new InvalidOperationException("Network is already configured.");

            _primaryLayers.Insert(index, layer);
        }

        /// <summary>
        /// Set's the <see cref="Network"/> into the state so that it's structure can be altered.
        /// </summary>
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

        /// <summary>
        /// Reset's all the <see cref="ILayer"/>s that match the conditions defined by the specified
        /// predicate to it's untrained state.
        /// </summary>
        /// <param name="predicate">A predicate that dictate's which <see cref="ILayer"/>s are reset.</param>
        public void ResetAll(Predicate<ILayer> predicate)
        {
            foreach (var layer in _layers)
            {
                if (predicate(layer))
                    layer.Reset();
            }
        }

        /// <summary>
        /// Reset's the <see cref="ILayer"/> at the specified index to it's untrained state.
        /// </summary>
        /// <param name="index">The index of the <see cref="ILayer"/> to be reset.</param>
        public void ResetLayer(int index)
        {
            _layers[index].Reset();
        }

        /// <summary>
        /// Reset's every <see cref="ILayer"/> in the network to their untrained states.
        /// </summary>
        public virtual void ResetNetwork()
        {
            foreach (var layer in _layers)
            {
                layer.Reset();
            }
        }

        /// <summary>
        /// Serializes the <see cref="Network"/> and saves it to a json file.
        /// </summary>
        /// <param name="file">The path of the json file.</param>
        public virtual void SaveToFile(string file)
        {
            try
            {
                // create the directory the file will be written to if it doesn't already exist
                Directory.CreateDirectory(Path.GetDirectoryName(file)!);

                using (StreamWriter writer = new(file))
                {
                    using(JsonWriter writer2 = new JsonTextWriter(writer))
                    {
                        var serializer = new JsonSerializer();
                        serializer.TypeNameHandling = TypeNameHandling.Auto;
                        serializer.Formatting = Formatting.Indented;
                        serializer.Serialize(writer2, this);
                    }
                }
                
            }
            catch (System.Exception e)
            {
                Console.WriteLine("Error occured when trying to save data to file: " + file + "\n" + e.ToString());
            }
        }

        /// <summary>
        /// Set's the <see cref="ActivationPattern"/> for the <see cref="Network"/>.
        /// </summary>
        /// <param name="pattern">The <see cref="ActivationPattern"/> defining what <see cref="ISecondaryLayer"/>s to insert after
        /// every <see cref="IPrimaryLayer"/> that is not a <see cref="IStructuralLayer"/>.</param>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="Network"/> is not being configured.</exception>
        public void SetActivationPattern(ActivationPattern pattern)
        {
            if (_configured)
                throw new InvalidOperationException("Network is already configured.");

            _activationPattern = pattern;
        }

        /// <summary>
        /// Sets up the <see cref="Network"/> so that it is ready for the specified inputs.
        /// </summary>
        /// <param name="batchSize">The number of images processed in each batch.</param>
        /// <param name="inputWidth">The width of the images.</param>
        /// <param name="inputLength">The length of the images.</param>
        /// <param name="boolLabels">The number of bools used to label each image.</param>
        /// <param name="floatLabels">The number of floats used to label each image.</param>
        /// While <see cref="Network"/> is designed as a Conditional GAN, but it can function as a non-Conditional GAN
        /// by only useing one bool or float label whose value is always true/1.
        public virtual void StartUp(int maxBatchSize, int inputWidth, int inputLength, int boolLabels, int floatLabels, AdamHyperParameters hyperParameters, int inputChannels)
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

            _boolLabels = boolLabels;
            _floatLabels = floatLabels;
            _adamHyperParameters ??= hyperParameters;
            _inputChannels = inputChannels;
        }

        protected void InitializeLayers(ref Shape current, int maxBatchSize)
        {
            _startBuffers ??= new();
            _middleBuffers ??= new();

            IOBuffers inputBuffers = _startBuffers;
            IOBuffers outputBuffers = _middleBuffers;
            outputBuffers.OutputDimensionArea(current.Volume);

            foreach (var layer in _layers)
            {
                current = layer.Startup(current, inputBuffers, maxBatchSize);
                if(layer is WeightedLayer weighted)
                {
                    foreach(var weight in weighted.SetUpWeights())
                        _weights.Add(weight);
                }
                else if(layer is BatchNormalization bn)
                {
                    bn.SetHyperParameters(_adamHyperParameters);
                }
                if (layer is not IUnchangedLayer)
                {
                    (inputBuffers, outputBuffers) = (outputBuffers, inputBuffers);
                }
            }
            _endBuffers = outputBuffers;
        }

        public void SetStartBuffers(Network network)
        {
            _startBuffers = network._endBuffers == network._middleBuffers ? network._startBuffers : network._middleBuffers; 
            _middleBuffers = network._endBuffers;
        }
    }
}