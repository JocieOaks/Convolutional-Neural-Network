using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers;
using Newtonsoft.Json;
using ILGPU;
using ConvolutionalNeuralNetwork.Layers.Serial;
using ConvolutionalNeuralNetwork.Layers.Initializers;
using ConvolutionalNeuralNetwork.Layers.Loss;
using System;
using System.Text.Json.Serialization;

namespace ConvolutionalNeuralNetwork
{
    /// <summary>
    /// The <see cref="Network"/> class is the base class for all Convolutional Neural Networks.
    /// </summary>
    public class Network : Loss
    {
        protected const bool PRINTSTOPWATCH = false;

        [JsonProperty] protected AdamHyperParameters _adamHyperParameters;
        [JsonProperty] protected bool _initialized = false;
        protected List<InputLayer> _inputLayers = new();
        [JsonProperty] protected TensorShape _inputShape;
        [Newtonsoft.Json.JsonIgnore] protected TensorShape _outputShape;
        [JsonProperty] protected List<int> _layerIndeces = new();
        protected bool _ready = false;
        [JsonProperty] protected List<ISerial> _serializedLayers = new();
        protected List<Weights> _weights;
        private readonly List<ILayer> _layers = new();
        protected delegate float LossFunction(Vector[] expected);

        [Newtonsoft.Json.JsonIgnore] public ArrayView<float> InGradient => _layers.Last().InGradient;
        [Newtonsoft.Json.JsonIgnore] public ArrayView<float> Input => _layers.First().Input;

        [Newtonsoft.Json.JsonIgnore] public ArrayView<float> OutGradient => _layers.First().OutGradient;
        [Newtonsoft.Json.JsonIgnore] public ArrayView<float> Output => _layers.Last().Output;
        /// <value>The number of <see cref="Layer"/>s in the <see cref="Network"/>.</value>
        protected int Depth => _layers.Count;

        [Newtonsoft.Json.JsonIgnore] protected Loss Loss { get; }

        public Network(Loss loss)
        {
            Loss = loss;
        }

        [Newtonsoft.Json.JsonConstructor] private Network(float gradientClip)
        {
            _gradientClip = gradientClip;
        }

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

        public SerialAugmentation AddAugmentation(Augmentation augmentationType)
        {
            SerialAugmentation augmentation = _serializedLayers.FirstOrDefault(x => x is SerialAugmentation act && act.Augmentation == augmentationType) as SerialAugmentation;
            augmentation ??= new SerialAugmentation()
            {
                Augmentation = augmentationType
            };

            AddSerialLayer(augmentation);
            return augmentation;
        }

        public SerialAvgPool AddAveragePool(int scale)
        {
            SerialAvgPool avgPool = _serializedLayers.FirstOrDefault(x => x is SerialAvgPool pool && pool.Scale == scale) as SerialAvgPool;
            avgPool ??= new SerialAvgPool(scale);
            AddSerialLayer(avgPool);
            return avgPool;
        }

        public SerialConcat AddConcatenation(SerialFork fork)
        {
            SerialConcat concatenation = new(fork);
            AddSerialLayer(concatenation);
            return concatenation;
        }

        public SerialConv AddConvolution(int outputDimensions, int filterSize, int stride = 1, IWeightInitializer initializer = null, bool useBias = true, IWeightInitializer biasInitializer = null, Activation activation = Activation.None)
        {
            Weights weights = new(initializer);
            Weights bias = null;
            if (useBias)
            {
                bias = new Weights(biasInitializer ?? new Constant(0));
            }

            SerialConv convolution = new(outputDimensions, filterSize, stride, weights, bias);
            AddSerialLayer(convolution);

            if (activation != Activation.None)
            {
                AddActivation(activation);
            }

            return convolution;
        }

        public SerialTransConv AddTransConv(int outputDimensions, int filterSize, int stride = 1, IWeightInitializer initializer = null, bool useBias = true, IWeightInitializer biasInitializer = null, Activation activation = Activation.None)
        {
            Weights weights = new(initializer);
            Weights bias = null;
            if (useBias)
            {
                bias = new Weights(biasInitializer ?? new Constant(0));
            }

            SerialTransConv transConv = new(outputDimensions, filterSize, stride, weights, bias);
            AddSerialLayer(transConv);

            if (activation != Activation.None)
            {
                AddActivation(activation);
            }

            return transConv;
        }

        public SerialDense AddDense(int outputUnits, IWeightInitializer initializer = null, bool useBias = true, IWeightInitializer biasInitializer = null, Activation activation = Activation.None)
        {
            Weights weights = new(initializer);
            Weights bias = null;
            if (useBias)
            {
                bias = new Weights(biasInitializer ?? new Constant(0));
            }

            SerialDense dense = new(outputUnits, weights, bias);
            AddSerialLayer(dense);

            if (activation != Activation.None)
            {
                AddActivation(activation);
            }

            return dense;
        }

        public SerialFork AddFork()
        {
            SerialFork fork = new();
            AddSerialLayer(fork);
            return fork;
        }

        public SerialInput AddInput(TensorShape input)
        {
            SerialInput layer = new(input);
            AddSerialLayer(layer);
            return layer;
        }

        public SerialReshape AddReshape(TensorShape output)
        {
            SerialReshape reshape = new(output);
            AddSerialLayer(reshape);
            return reshape;
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

        public SerialSum AddSummation(int outputDimensions)
        {
            SerialSum summation = _serializedLayers.FirstOrDefault(x => x is SerialSum sum && sum.OutputDimensions == outputDimensions) as SerialSum;
            summation ??= new SerialSum(outputDimensions);
            AddSerialLayer(summation);
            return summation;
        }

        public SerialUp AddUpsampling(int scale)
        {
            SerialUp upsampling = _serializedLayers.FirstOrDefault(x => x is SerialUp up && up.Scale == scale) as SerialUp;
            upsampling ??= new SerialUp(scale);
            AddSerialLayer(upsampling);
            return upsampling;
        }

        public SerialBatchNorm AddBatchNormalization()
        {
            SerialBatchNorm norm = new();
            AddSerialLayer(norm);
            return norm;
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
            if (_initialized)
                return;

            TensorShape shape = new();

            foreach (var index in _layerIndeces)
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

            TensorShape shape = new();
            InitializeLayers(ref shape, maxBatchSize);

            _adamHyperParameters ??= hyperParameters;
            _ready = true;
        }

        public override void Startup(PairedBuffers buffers, TensorShape outputShape, int maxBatchSize)
        {
            Buffers = buffers.Compliment;
        }

        public (float, float) Test(List<Tensor[]> inputs, Vector[] expected, bool saveOutput)
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

            if (saveOutput)
            {
                for (int i = 0; i < _outputs.GetLength(0); i++)
                {
                    _outputs[i].SyncCPU(Output.SubView(i * _inputShape.Volume, _inputShape.Volume));
                }
            }

            return Loss.GetLoss(expected);
        }

        protected Tensor[] _outputs;

        [Newtonsoft.Json.JsonIgnore]
        public Tensor[] GetOutputs => _outputs;

        public (float, float) Train(List<Tensor[]> inputs, Vector[] expected, bool update = true)
        {

            if (inputs.Count != _inputLayers.Count)
            {
                throw new ArgumentException("Incorrect input count.");
            }

            for (int i = 0; i < inputs.Count; i++)
            {
                _inputLayers[i].SetInput(inputs[i]);
            }

            return Train(expected, false, update);
        }

        [JsonProperty] private float _gradientClip;

        public (float, float) Train(Vector[] expected, bool skipInputLayers, bool update)
        {
            int batchSize = expected.Length;
            for (int i = 0; i < Depth; i++)
            {
                if (!skipInputLayers || _layers[i] is not InputLayer)
                    Utility.StopWatch(() => _layers[i].Forward(batchSize), $"Forwards {i} {_layers[i].Name}", PRINTSTOPWATCH);
            }

            (float, float) loss = Loss.GetLoss(expected);

            for (int j = Depth - 1; j >= 0; j--)
            {
                Utility.StopWatch(() => _layers[j].Backwards(batchSize, update), $"Backwards {j} {_layers[j].Name}", PRINTSTOPWATCH);
            }

            if (update)
            {
                _adamHyperParameters.Update();
                foreach (var weight in _weights)
                {
                    weight.UpdateWeights(_adamHyperParameters);
                }
            }
            return loss;
        }

        public void Generate(List<Tensor[]> inputs, bool saveOutput)
        {
            if (inputs.Count != _inputLayers.Count)
            {
                throw new ArgumentException("Incorrect input count.");
            }

            int batchSize = inputs[0].Length;

            for (int i = 0; i < inputs.Count; i++)
            {
                _inputLayers[i].SetInput(inputs[i]);
            }

            for (int i = 0; i < Depth; i++)
            {
                Utility.StopWatch(() => _layers[i].Forward(batchSize), $"Forwards {i} {_layers[i].Name}", PRINTSTOPWATCH);
            }

            if (saveOutput)
            {
                for (int i = 0; i < _outputs.GetLength(0); i++)
                {
                    _outputs[i].SyncCPU(Output.SubView(i * _outputShape.Volume, _outputShape.Volume));
                }
            }
        }

        protected void InitializeLayers(ref TensorShape current, int maxBatchSize)
        {
            Buffers ??= new();

            PairedBuffers inputBuffers = Buffers;
            PairedBuffers outputBuffers = Buffers.Compliment ?? new();
            PairedBuffers.SetCompliment(inputBuffers, outputBuffers);
            outputBuffers.OutputDimensionArea(current.Volume);

            foreach (var layer in _layers)
            {
                current = layer.Startup(current, inputBuffers, maxBatchSize);
                if (layer is not IReflexiveLayer)
                {
                    (inputBuffers, outputBuffers) = (outputBuffers, inputBuffers);
                }
            }

            Loss?.Startup(outputBuffers, current, maxBatchSize);
            inputBuffers.Allocate(maxBatchSize);
            outputBuffers.Allocate(maxBatchSize);

            _outputShape = current;
            _outputs = new Tensor[maxBatchSize];
            for (int i = 0; i < maxBatchSize; i++)
            {
                _outputs[i] = new Tensor(current);
            }
        }

        private void Construct()
        {
            if (!_initialized)
            {
                throw new InvalidOperationException("Network has not been initialized.");
            }

            foreach (var index in _layerIndeces)
            {
                var layer = _serializedLayers[index].Construct();
                if (layer is InputLayer input)
                {
                    _inputLayers.Add(input);
                }
                _layers.Add(layer);
            }
        }

        public override (float, float) GetLoss(Vector[] groundTruth)
        {
            return Train(groundTruth, true, false);
        }
    }
}