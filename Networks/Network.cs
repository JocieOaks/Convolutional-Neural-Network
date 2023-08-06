using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Design;
using ConvolutionalNeuralNetwork.Layers;
using Newtonsoft.Json;
using ILGPU;
using ConvolutionalNeuralNetwork.Layers.Weighted;
using ConvolutionalNeuralNetwork.Layers.Serial;
using ConvolutionalNeuralNetwork.Layers.Initializers;
using ILGPU.Algorithms.ScanReduceOperations;

namespace ConvolutionalNeuralNetwork.Networks
{
    /// <summary>
    /// The <see cref="Network"/> class is the base class for all Convolutional Neural Networks.
    /// </summary>
    public abstract class Network
    {
        protected const bool PRINTSTOPWATCH = false;

        [JsonProperty] protected AdamHyperParameters _adamHyperParameters;
        protected IOBuffers _endBuffers;
        [JsonProperty] protected bool _initialized = false;
        protected List<InputLayer> _inputLayers = new();
        [JsonProperty] protected Shape _inputShape;
        [JsonProperty] protected List<int> _layerIndeces = new();
        protected IOBuffers _middleBuffers;
        protected bool _ready = false;
        [JsonProperty] protected List<ISerial> _serializedLayers = new();
        protected IOBuffers _startBuffers;
        protected List<Weights> _weights;
        private readonly List<ILayer> _layers = new();
        protected delegate float LossFunction(Vector[] expected);

        [JsonIgnore] public ArrayView<float> InGradient => _layers.Last().InGradient;
        [JsonIgnore] public ArrayView<float> Input => _layers.First().Input;

        [JsonIgnore] public ArrayView<float> OutGradient => _layers.First().OutGradient;
        [JsonIgnore] public ArrayView<float> Output => _layers.Last().Output;
        /// <value>The number of <see cref="Layer"/>s in the <see cref="Network"/>.</value>
        protected int Depth => _layers.Count;

        [JsonIgnore] protected virtual LossFunction Loss { get; }

        public SerialActivation AddActivation(Activation activationType)
        {
            SerialActivation activation = _serializedLayers.FirstOrDefault(x => x is SerialActivation act && act.Activation == activationType) as SerialActivation;
            activation ??= new SerialActivation()
            {
                Activation = activationType
            };

            AddSerialLayer(activation);
            return activation;
        }

        public SerialAveragePool AddAveragePool(int scale)
        {
            SerialAveragePool avgPool = _serializedLayers.FirstOrDefault(x => x is SerialAveragePool pool && pool.Scale == scale) as SerialAveragePool;
            avgPool ??= new SerialAveragePool(scale);
            AddSerialLayer(avgPool);
            return avgPool;
        }

        public SerialConcatenate AddConcatenation(SerialFork fork)
        {
            SerialConcatenate concatenation = new(fork);
            AddSerialLayer(concatenation);
            return concatenation;
        }

        public SerialConvolution AddConvolution(int outputDimensions, int filterSize, int stride = 1, IWeightInitializer initializer = null, bool useBias = true, IWeightInitializer biasInitializer = null, Activation activation = Activation.None)
        {
            Weights weights = new (initializer);
            Weights bias = null;
            if(useBias)
            {
                bias = new Weights(biasInitializer ?? new Constant(0));
            }

            SerialConvolution convolution = new(outputDimensions, filterSize, stride, weights, bias);
            AddSerialLayer(convolution);

            if(activation != Activation.None)
            {
                AddActivation(activation);
            }

            return convolution;
        }

        public SerialFork AddFork()
        {
            SerialFork fork = new();
            AddSerialLayer(fork);
            return fork;
        }

        public SerialInput AddInput(Shape input)
        {
            SerialInput layer = new(input);
            AddSerialLayer(layer);
            return layer;
        }

        public void AddSerialLayer(ISerial layer)
        {
            if (!_serializedLayers.Contains(layer))
            {
                _serializedLayers.Add(layer);
            }

            _layerIndeces.Add(_serializedLayers.IndexOf(layer));
        }

        public SerialOut AddSkipOut(SerialFork fork)
        {
            SerialOut skipOut = new(fork);
            AddSerialLayer(skipOut);
            return skipOut;
        }

        public SerialSummation AddSummation(int outputDimensions)
        {
            SerialSummation summation = _serializedLayers.FirstOrDefault(x => x is SerialSummation sum && sum.OutputDimensions == outputDimensions) as SerialSummation;
            summation ??= new SerialSummation(outputDimensions);
            AddSerialLayer(summation);
            return summation;
        }

        public SerialUpsampling AddUpsampling(int scale)
        {
            SerialUpsampling upsampling = _serializedLayers.FirstOrDefault(x => x is SerialUpsampling up && up.Scale == scale) as SerialUpsampling;
            upsampling ??= new SerialUpsampling(scale);
            AddSerialLayer(upsampling);
            return upsampling;
        }

        public SerialWarp AddWarp()
        {
            SerialWarp warp = _serializedLayers.FirstOrDefault(x => x is SerialWarp) as SerialWarp;
            warp ??= new SerialWarp();
            AddSerialLayer(warp); 
            return warp;
        }
        public void Initialize()
        {
            if(_initialized)
                return;

            Shape shape = new();

            foreach(var index in _layerIndeces)
            {
                shape = _serializedLayers[index].Initialize(shape);
            }

            _initialized = true;
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
                    using (JsonWriter writer2 = new JsonTextWriter(writer))
                    {
                        var serializer = new JsonSerializer();
                        serializer.TypeNameHandling = TypeNameHandling.Auto;
                        serializer.Formatting = Formatting.Indented;
                        serializer.Serialize(writer2, this);
                    }
                }

            }
            catch (Exception e)
            {
                Console.WriteLine("Error occured when trying to save data to file: " + file + "\n" + e.ToString());
            }
        }

        public void SetStartBuffers(Network network)
        {
            _startBuffers = network._endBuffers == network._middleBuffers ? network._startBuffers : network._middleBuffers;
            _middleBuffers = network._endBuffers;
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
        public virtual void StartUp(int maxBatchSize, AdamHyperParameters hyperParameters)
        {
            Construct();

            _weights = new();
            foreach (var layer in _serializedLayers)
            {
                if (layer is SerialWeighted weightedLayer)
                {
                    weightedLayer.GetWeights(_weights);
                }
            }

            Shape shape = new();
            InitializeLayers(ref shape, maxBatchSize);

            _startBuffers.Allocate(maxBatchSize);
            _middleBuffers.Allocate(maxBatchSize);
            IOBuffers.SetCompliment(_startBuffers, _middleBuffers);

            _adamHyperParameters ??= hyperParameters;
            _ready = true;
        }

        public float Test(List<FeatureMap[][]> inputs, Vector[] expected)
        {
            int batchSize = inputs[0].Length;
            for (int i = 0; i < inputs.Count; i++)
            {
                _inputLayers[i].SetInput(inputs[i]);
            }

            for (int i = 0; i < Depth; i++)
            {
                Utility.StopWatch(() => _layers[i].Forward(batchSize), $"Forwards {i} {_layers[i].Name}", PRINTSTOPWATCH);
            }

            return Loss(expected);
        }

        public float Train(List<FeatureMap[][]> inputs, Vector[] expected)
        {
            int batchSize = inputs[0].Length;
            for (int i = 0; i < inputs.Count; i++)
            {
                _inputLayers[i].SetInput(inputs[i]);
            }

            for (int i = 0; i < Depth; i++)
            {
                Utility.StopWatch(() => _layers[i].Forward(batchSize), $"Forwards {i} {_layers[i].Name}", PRINTSTOPWATCH);
            }

            float loss = Loss(expected);

            for (int j = Depth - 1; j >= 0; j--)
            {
                Utility.StopWatch(() => _layers[j].Backwards(batchSize, true), $"Backwards {j} {_layers[j].Name}", PRINTSTOPWATCH);
            }

            _adamHyperParameters.Update();
            foreach (var weight in _weights)
            {
                weight.UpdateWeights(_adamHyperParameters);
            }

            return loss;
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
                if (layer is BatchNormalization bn)
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

        private void Construct()
        {
            if(!_initialized)
            {
                throw new InvalidOperationException("Network has not been initialized.");
            }

            foreach(var index in _layerIndeces)
            {
                var layer = _serializedLayers[index].Construct();
                if(layer is InputLayer input)
                {
                    _inputLayers.Add(input);
                }
                _layers.Add(layer);
            }
        }
    }
}