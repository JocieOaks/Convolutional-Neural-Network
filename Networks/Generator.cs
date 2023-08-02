using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers;
using Newtonsoft.Json;
using ConvolutionalNeuralNetwork.Layers.Activations;

namespace ConvolutionalNeuralNetwork.Networks
{
    /// <summary>
    /// The <see cref="Generator"/> is a <see cref="Network"/> for generating images from a starting image.
    /// </summary>
    public class Generator : Network
    {
        private FeatureMap[,] _outputs;
        private readonly int _latentDimensions = 100;
        private int _outputArea;

        /// <summary>
        /// Loads a <see cref="Generator"/> from a json file.
        /// </summary>
        /// <param name="file">The path of the json file.</param>
        /// <returns>Returns the deserialized <see cref="Generator"/>.</returns>
        public static Generator LoadFromFile(string file)
        {
            Generator generator = null;

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
                            generator = serializer.Deserialize<Generator>(reader);
                        }
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine("Error occured when trying to load data from file: " + file + "\n" + e.ToString());
                }
            }

            return generator;
        }

        /// <summary>
        /// Backpropogates through the network, updating every <see cref="ILayer"/> in the <see cref="Generator"/>.
        /// </summary>
        /// <param name="gradients">An array of <see cref="FeatureMap"/>'s containing the gradients for the last layer of the <see cref="Generator"/>.</param>
        /// <param name="inputs">The images and their associatd labels for this iteration of training.</param>
        /// <param name="learningRate">The learning rate defining the degree to which each layer should be updated.</param>
        public void Backwards(int batchSize)
        {
            _adamHyperParameters.Update();

            for (int i = Depth - 1; i >= 0; i--)
            {
                Utility.StopWatch(() => _layers[i].Backwards(batchSize, true), $"Backwards {i} {_layers[i].Name}", PRINTSTOPWATCH);
            }
        }

        /// <summary>
        /// Forward propagates through the network to generate a new <see cref="FeatureMap"/> image from the starting image.
        /// </summary>
        /// <param name="input">The images and their asscoiated labels.</param>
        /// <param name="inference">Determines whether the <see cref="Generator"/> is training or inferring. Defaults to false.</param>
        public void Forward(ImageInput[] input)
        {
            int batchSize = input.Length;

            for(int i = 0; i < batchSize; i++)
            {
                Vector latentVector = input[i].LabelVector(_latentDimensions);
                latentVector.CopyToBuffer(Input.SubView(i * latentVector.Length, latentVector.Length));
            }

            for (int i = 0; i < Depth; i++)
            {
                Utility.StopWatch(() => _layers[i].Forward(batchSize), $"Forwards {i} {_layers[i].Name}", PRINTSTOPWATCH);
            }
        }

        public FeatureMap[,] GetFeatureMaps(int batchSize)
        {
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < _inputChannels; j++)
                { 
                    _outputs[i,j].SyncCPU(Output.SubView(_outputArea * (i * _inputChannels + j), _outputArea));
                }
            }

            return _outputs;
        }

        /// <inheritdoc/>
        public override void StartUp(int maxBatchSize, int width, int length, int labelBools, int labelFloats, AdamHyperParameters hyperParameters, int inputChannels)
        {
            base.StartUp(maxBatchSize, width, length, labelBools, labelFloats, hyperParameters, inputChannels);

            for(int i = _layers.Count - 1; i >= 0; i--)
            {
                if (_layers[i] is IPrimaryLayer && _layers[i] is not IStructuralLayer)
                {
                    break;
                }
                if (_layers[i] is ISecondaryLayer)
                    _layers.Remove(_layers[i]);
            }

            _layers.Add(new HyperTan());

            _outputs = new FeatureMap[maxBatchSize, inputChannels];
            _outputArea = width * length;

            for (int i = 0; i < maxBatchSize; i++)
            {
                for (int j = 0; j < inputChannels; j++)
                {
                    _outputs[i, j] = new FeatureMap(width, length);
                }
            }

            Shape inputShape = new(1, 1, _latentDimensions + LabelCount);

            InitializeLayers(ref inputShape, maxBatchSize);

            _ready = true;
        }
    }
}