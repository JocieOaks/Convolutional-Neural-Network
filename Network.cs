using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers;
using Newtonsoft.Json;
using ConvolutionalNeuralNetwork.Layers.Serial;
using ConvolutionalNeuralNetwork.Layers.Loss;
using ConvolutionalNeuralNetwork.DataTypes.Initializers;
using ConvolutionalNeuralNetwork.Layers.Serial.SkipConnection;
using ConvolutionalNeuralNetwork.Layers.Serial.Weighted;

namespace ConvolutionalNeuralNetwork
{
    /// <summary>
    /// The <see cref="Network"/> class is a collection of <see cref="Layer"/>s that can be trained to generate images.
    /// </summary>
    public class Network : Loss
    {
        private readonly List<Input> _inputLayers = new();
        private readonly List<Layer> _layers = new();
        [JsonProperty] private AdamHyperParameters _adamHyperParameters;
        [JsonProperty] private bool _initialized;
        [JsonProperty] private List<int> _layerIndices = new();
        private Loss _loss;
        private Tensor[] _outputs;
        private TensorShape _outputShape;
        [JsonProperty] private List<ISerialLayer> _serializedLayers = new();
        private List<Weights> _weights;

        /// <summary>
        /// Initializes a new instance of the <see cref="Network"/> class.
        /// </summary>
        /// <param name="loss">The <see cref="Layers.Loss.Loss"/> layer used to determine the loss and gradient of the <see cref="Network"/>.</param>
        public Network(Loss loss)
        {
            Loss = loss;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Network"/> class; used for deserialization.
        /// </summary>
        [JsonConstructor]
        private Network()
        {
        }

        /// <value>The number of <see cref="Layer"/>s in the <see cref="Network"/>.</value>
        private int Depth => _layers.Count;

        /// <value>The <see cref="Network"/>'s loss function that determines the loss value and <see cref="Layer"/> gradients.</value>
        [JsonIgnore]
        public Loss Loss
        {
            get => _loss;
            private set => _loss ??= value;
        }

        /// <summary>
        /// Appends an activation <see cref="Layer"/> to the end of the <see cref="Network"/>.
        /// </summary>
        /// <param name="activationType">The type of activation <see cref="Layer"/> to add.</param>
        /// <returns>Returns the <see cref="SerialActivation"/> of the added <see cref="Layer"/>.</returns>
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

        /// <summary>
        /// Appends an augmentation <see cref="Layer"/> to the end of the <see cref="Network"/>.
        /// </summary>
        /// <param name="augmentationType">The type of augmentation <see cref="Layer"/> to add.</param>
        /// <returns>Returns the <see cref="SerialAugmentation"/> of the added <see cref="Layer"/>.</returns>
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

        /// <summary>
        /// Appends an <see cref="AveragePool"/> layer to the end of the <see cref="Network"/>.
        /// </summary>
        /// <param name="scale">The scale of the pooling filter.</param>
        /// <returns>Returns the <see cref="SerialAvgPool"/> of the added <see cref="Layer"/>.</returns>
        public SerialAvgPool AddAveragePool(int scale)
        {
            SerialAvgPool avgPool = _serializedLayers.FirstOrDefault(x => x is SerialAvgPool pool && pool.Scale == scale) as SerialAvgPool;
            avgPool ??= new SerialAvgPool(scale);
            AddSerialLayer(avgPool);
            return avgPool;
        }

        /// <summary>
        /// Appends an <see cref="Layers.Weighted.BatchNormalization"/> layer to the end of the <see cref="Network"/>.
        /// </summary>
        /// <returns>Returns the <see cref="SerialBatchNorm"/> of the added <see cref="Layer"/>.</returns>
        public SerialBatchNorm AddBatchNormalization()
        {
            SerialBatchNorm norm = new();
            AddSerialLayer(norm);
            return norm;
        }

        /// <summary>
        /// Appends an <see cref="Layers.SkipConnection.Concatenate"/> layer to the end of the <see cref="Network"/>.
        /// </summary>
        /// <param name="fork">The <see cref="SerialFork"/> that connects to the <see cref="SerialConcat"/>.</param>
        /// <returns>Returns the <see cref="SerialConcat"/> of the added <see cref="Layer"/>.</returns>
        public SerialConcat AddConcatenation(SerialFork fork)
        {
            SerialConcat concatenation = new(fork);
            AddSerialLayer(concatenation);
            return concatenation;
        }

        /// <summary>
        /// Appends an <see cref="Layers.Weighted.Convolution"/> layer to the end of the <see cref="Network"/>.
        /// </summary>
        /// <param name="outputDimensions">The dimensions of the <see cref="Layer"/>'s output.</param>
        /// <param name="filterSize">The width and length of the <see cref="Layer"/>'s filters.</param>
        /// <param name="stride">The <see cref="Layer"/>'s horizontal and vertical stride.</param>
        /// <param name="initializer">The <see cref="IWeightInitializer"/> for the <see cref="Layer"/>'s filters.</param>
        /// <param name="useBias">Whether the see <see cref="Layer"/> should have a bias for the outputs.</param>
        /// <param name="biasInitializer">The <see cref="IWeightInitializer"/> for the <see cref="Layer"/>'s bias <see cref="Weights"/>.
        /// Defaults to 0's if left unspecified.</param>
        /// <param name="activation">The activation layer to use after this <see cref="Layer"/>; can be <see cref="Activation.None"/>.</param>
        /// <returns>Returns the <see cref="SerialConv"/> of the added <see cref="Layer"/>.</returns>
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

        /// <summary>
        /// Appends an <see cref="Layers.Weighted.Dense"/> layer to the end of the <see cref="Network"/>.
        /// </summary>
        /// <param name="outputUnits">The length of the <see cref="Layer"/>'s output <see cref="Vector"/>.</param>
        /// <param name="initializer">The <see cref="IWeightInitializer"/> for the <see cref="Layer"/>'s filters.</param>
        /// <param name="useBias">Whether the see <see cref="Layer"/> should have a bias for the outputs.</param>
        /// <param name="biasInitializer">The <see cref="IWeightInitializer"/> for the <see cref="Layer"/>'s bias <see cref="Weights"/>.
        /// Defaults to 0's if left unspecified.</param>
        /// <param name="activation">The activation layer to use after this <see cref="Layer"/>; can be <see cref="Activation.None"/>.</param>
        /// <returns>Returns the <see cref="SerialDense"/> of the added <see cref="Layer"/>.</returns>
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

        /// <summary>
        /// Appends an <see cref="Layers.SkipConnection.Fork"/> layer to the end of the <see cref="Network"/>.
        /// </summary>
        /// <returns>Returns the <see cref="SerialFork"/> of the added <see cref="Layer"/>.</returns>
        public SerialFork AddFork()
        {
            SerialFork fork = new();
            AddSerialLayer(fork);
            return fork;
        }

        /// <summary>
        /// Appends an <see cref="Input"/> layer to the end of the <see cref="Network"/>.
        /// </summary>
        /// <param name="input">The <see cref="TensorShape"/> of the <see cref="Network"/>'s input <see cref="Tensor"/>.</param>
        /// <returns>Returns the <see cref="SerialInput"/> of the added <see cref="Layer"/>.</returns>
        public SerialInput AddInput(TensorShape input)
        {
            SerialInput layer = new(input);
            AddSerialLayer(layer);
            return layer;
        }

        /// <summary>
        /// Appends an <see cref="Reshape"/> layer to the end of the <see cref="Network"/>.
        /// </summary>
        /// <param name="output">The <see cref="TensorShape"/> to reform the previous <see cref="Layer"/>'s output <see cref="Tensor"/> into.</param>
        /// <returns>Returns the <see cref="SerialReshape"/> of the added <see cref="Layer"/>.</returns>
        public SerialReshape AddReshape(TensorShape output)
        {
            SerialReshape reshape = new(output);
            AddSerialLayer(reshape);
            return reshape;
        }

        /// <summary>
        /// Appends a layer to the end of the <see cref="Network"/> from the given <see cref="ISerialLayer"/>.
        /// Allows for a <see cref="Layer"/> to be used multiple times in a <see cref="Network"/>.
        /// </summary>
        /// <param name="layer">The <see cref="ISerialLayer"/> corresponding to the <see cref="Layer"/> being added.</param>
        public void AddSerialLayer(ISerialLayer layer)
        {
            if (!_serializedLayers.Contains(layer))
            {
                _serializedLayers.Add(layer);
            }

            _layerIndices.Add(_serializedLayers.IndexOf(layer));
        }

        /// <summary>
        /// Appends an <see cref="Layers.SkipConnection.Out"/> layer to the end of the <see cref="Network"/>.
        /// </summary>
        /// <param name="fork">The <see cref="SerialFork"/> that connects to the <see cref="SerialOut"/>.</param>
        /// <returns>Returns the <see cref="SerialOut"/> of the added <see cref="Layer"/>.</returns>
        public SerialOut AddSkipOut(SerialFork fork)
        {
            SerialOut skipOut = new(fork);
            AddSerialLayer(skipOut);
            return skipOut;
        }

        /// <summary>
        /// Appends an <see cref="Summation"/> layer to the end of the <see cref="Network"/>.
        /// </summary>
        /// <param name="outputDimensions">The number of dimensions in the <see cref="Layer"/>'s output <see cref="Tensor"/>.</param>
        /// <returns>Returns the <see cref="SerialSum"/> of the added <see cref="Layer"/>.</returns>
        public SerialSum AddSummation(int outputDimensions)
        {
            SerialSum summation = _serializedLayers.FirstOrDefault(x => x is SerialSum sum && sum.OutputDimensions == outputDimensions) as SerialSum;
            summation ??= new SerialSum(outputDimensions);
            AddSerialLayer(summation);
            return summation;
        }

        /// <summary>
        /// Appends an <see cref="Layers.Weighted.TransposeConvolution"/> layer to the end of the <see cref="Network"/>.
        /// </summary>
        /// <param name="outputDimensions">The dimensions of the <see cref="Layer"/>'s output.</param>
        /// <param name="filterSize">The width and length of the <see cref="Layer"/>'s filters.</param>
        /// <param name="stride">The <see cref="Layer"/>'s horizontal and vertical stride.</param>
        /// <param name="initializer">The <see cref="IWeightInitializer"/> for the <see cref="Layer"/>'s filters.</param>
        /// <param name="useBias">Whether the see <see cref="Layer"/> should have a bias for the outputs.</param>
        /// <param name="biasInitializer">The <see cref="IWeightInitializer"/> for the <see cref="Layer"/>'s bias <see cref="Weights"/>.
        /// Defaults to 0's if left unspecified.</param>
        /// <param name="activation">The activation layer to use after this <see cref="Layer"/>; can be <see cref="Activation.None"/>.</param>
        /// <returns>Returns the <see cref="SerialTransConv"/> of the added <see cref="Layer"/>.</returns>
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

        /// <summary>
        /// Appends an <see cref="Upsampling"/> layer to the end of the <see cref="Network"/>.
        /// </summary>
        /// <param name="scale">The amount to scale the <see cref="Layer"/>'s input <see cref="Tensor"/> by.</param>
        /// <returns>Returns the <see cref="SerialUp"/> of the added <see cref="Layer"/>.</returns>
        public SerialUp AddUpsampling(int scale)
        {
            SerialUp upsampling = _serializedLayers.FirstOrDefault(x => x is SerialUp up && up.Scale == scale) as SerialUp;
            upsampling ??= new SerialUp(scale);
            AddSerialLayer(upsampling);
            return upsampling;
        }

        /// <summary>
        /// Generates an output <see cref="Tensor"/> from the <see cref="Network"/> without calculating gradients or back-propagating.
        /// </summary>
        /// <param name="inputs">A list of <see cref="Tensor"/>s to generate from, typically a latent vector.</param>
        /// <param name="saveOutput">Determines whether the generated <see cref="Tensor"/>s should be saved to the CPU.</param>
        /// <returns>Returns the generated <see cref="Tensor"/>s if they were saved to the CPU, null otherwise.</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="inputs"/> is not the correct length for the number of <see cref="Network"/> inputs.</exception>
        public Tensor[] Generate(List<Tensor[]> inputs, bool saveOutput)
        {
            if (inputs.Count != _inputLayers.Count)
            {
                throw new ArgumentException("Incorrect input count.");
            }

            int batchSize = inputs[0].Length;
            //Copy inputs
            for (int i = 0; i < inputs.Count; i++)
            {
                _inputLayers[i].SetInput(inputs[i]);
            }

            //Forward pass
            for (int i = 0; i < Depth; i++)
            {
                _layers[i].Forward(batchSize);
            }

            //Copying generated output from the GPU.
            if (saveOutput)
            {
                for (int i = 0; i < _outputs.GetLength(0); i++)
                {
                    _outputs[i].SyncCPU(_layers.Last().Output.SubView(i * _outputShape.Volume, _outputShape.Volume));
                }

                return _outputs;
            }

            return null;
        }

        /// <inheritdoc />
        public override (float, float) GetLoss(Vector[] groundTruth)
        {
            return Train(groundTruth, true, false);
        }

        /// <summary>
        /// Initializes the <see cref="ISerialLayer"/> layers and their <see cref="Weights"/>.
        /// </summary>
        public void Initialize()
        {
            if (_initialized)
                return;

            TensorShape shape = new();

            foreach (var index in _layerIndices)
            {
                shape = _serializedLayers[index].Initialize(shape);
            }

            _initialized = true;
        }

        /// <summary>
        /// Loads a <see cref="Network"/> from the given Json file.
        /// </summary>
        /// <param name="file">The file containing a serialized <see cref="Network"/>.</param>
        /// <param name="loss">The <see cref="Layers.Loss.Loss"/> for the deserialized <see cref="Network"/>.</param>
        /// <returns>Returns the deserialized <see cref="Network"/>.</returns>
        public static Network LoadFromFile(string file, Loss loss)
        {
            Network network = null;

            if (File.Exists(file))
            {
                try
                {
                    string dataToLoad;
                    using (FileStream stream = new(file, FileMode.Open))
                    {
                        using StreamReader read = new(stream);
                        dataToLoad = read.ReadToEnd();
                    }
                    network = JsonConvert.DeserializeObject<Network>(dataToLoad, new JsonSerializerSettings
                    {
                        TypeNameHandling = TypeNameHandling.Auto
                    });

                    if (network != null)
                    {
                        network.Loss = loss;
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine("Error occurred when trying to load data from file: " + file + "\n" + e);
                }
            }

            return network;
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

                using StreamWriter writer = new(file);
                using JsonWriter writer2 = new JsonTextWriter(writer);
                var serializer = new JsonSerializer
                {
                    TypeNameHandling = TypeNameHandling.Auto,
                    Formatting = Formatting.Indented
                };
                serializer.Serialize(writer2, this);
            }
            catch (Exception e)
            {
                Console.WriteLine("Error occurred when trying to save data to file: " + file + "\n" + e);
            }
        }

        /// <inheritdoc />
        public override void Startup(PairedGPUViews views, TensorShape outputShape, int maxBatchSize)
        {
            Views = views.Compliment;
        }

        /// <summary>
        /// Sets up the <see cref="Network"/> so that it is ready for initial use.
        /// </summary>
        /// <param name="maxBatchSize">The maximum number of images processed in each batch.</param>
        /// <param name="hyperParameters">The <see cref="AdamHyperParameters"/> specifying how <see cref="Weights"/> are updated.</param>
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
        }

        /// <summary>
        /// Trains the <see cref="Network"/> with the specified inputs.
        /// </summary>
        /// <param name="inputs">A list of <see cref="Tensor"/> arrays used to train the <see cref="Network"/>.
        /// Each element of the list corresponds with an element of the training batch.</param>
        /// <param name="expected">An array of <see cref="Vector"/>s that represent the expected outputs of the <see cref="Network"/>.</param>
        /// <param name="update">Determines whether the <see cref="Network"/> should be updated on the backwards pass; defaults to true.</param>
        /// <returns>Returns a tuple containing the loss and accuracy of the training batch.</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="inputs"/> is not the correct length for the number of <see cref="Network"/> inputs.</exception>
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

        /// <summary>
        /// Trains the <see cref="Network"/> using whatever inputs have already been set.
        /// Typically used in a GAN, so that the discriminator <see cref="Network"/> is trained using
        /// the generated outputs of the generator <see cref="Network"/>.
        /// </summary>
        /// <param name="expected">An array of <see cref="Vector"/>s that represent the expected outputs of the <see cref="Network"/>.</param>
        /// <param name="update">Determines whether the <see cref="Network"/> should be updated on the backwards pass; defaults to true.</param>
        /// <returns>Returns a tuple containing the loss and accuracy of the training batch.</returns>
        public (float, float) Train(Vector[] expected, bool update = true)
        {
            return Train(expected, true, update);
        }

        private (float, float) Train(Vector[] expected, bool skipInputLayers, bool update)
        {
            int batchSize = expected.Length;

            //Forward pass
            for (int i = 0; i < Depth; i++)
            {
                if (!skipInputLayers || _layers[i] is not Input)
                    _layers[i].Forward(batchSize);
            }

            //Calculate loss and gradient
            (float, float) loss = Loss.GetLoss(expected);

            //Backwards pass
            for (int j = Depth - 1; j >= 0; j--)
            {
                _layers[j].Backwards(batchSize, update);
            }

            //Update Weights
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


        private void Construct()
        {
            if (!_initialized)
            {
                throw new InvalidOperationException("Network has not been initialized.");
            }

            //Construct usable layers from the serialized layers
            foreach (var index in _layerIndices)
            {
                var layer = _serializedLayers[index].Construct();
                if (layer is Input input)
                {
                    _inputLayers.Add(input);
                }
                _layers.Add(layer);
            }
        }

        private void InitializeLayers(ref TensorShape current, int maxBatchSize)
        {
            Views ??= new();

            PairedGPUViews inputViews = Views;
            PairedGPUViews outputViews = Views.Compliment ?? new();
            PairedGPUViews.SetCompliment(inputViews, outputViews);
            outputViews.OutputDimensionArea(current.Volume);

            foreach (var layer in _layers)
            {
                current = layer.Startup(current, inputViews, maxBatchSize);
                //Each layer uses the output of the previous layer as its input, so output and input are swapped between layers.
                //The exception is reflexive layers modify their input layer without copying it to the output, so the input for the following layer
                //will be the same as the input for the reflexive layer.
                if (!layer.Reflexive)                                       
                {
                    (inputViews, outputViews) = (outputViews, inputViews);
                }
            }

            Loss?.Startup(outputViews, current, maxBatchSize);

            //Allocate the required space on the GPU for the memory buffers.
            inputViews.Allocate(maxBatchSize);
            outputViews.Allocate(maxBatchSize);

            _outputShape = current;
            _outputs = new Tensor[maxBatchSize];
            for (int i = 0; i < maxBatchSize; i++)
            {
                _outputs[i] = new Tensor(current);
            }
        }
    }
}