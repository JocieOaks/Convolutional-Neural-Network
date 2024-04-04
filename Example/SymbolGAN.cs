using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.DataTypes.Initializers;
using ConvolutionalNeuralNetwork.Layers.Loss;
using ConvolutionalNeuralNetwork.Layers.Serial;

namespace ConvolutionalNeuralNetwork.Example
{
    /// <summary>
    /// The <see cref="SymbolGAN"/> class is GAN for replicating a specific handwritten character.
    /// Built for the MNIST data set.
    /// </summary>
    internal class SymbolGAN
    {
        private readonly int _batchSize;
        private readonly Vector _fakeClassifications;
        private readonly Tensor[] _fakeInputs;
        private readonly Vector _generatorClassifications;
        private readonly Tensor[] _generatorInputs;
        private readonly int _imageLength;
        private readonly int _imageWidth;
        private readonly int _latentDimensions;

        private readonly Vector _realClassifications;
        private readonly Tensor[] _realInputs;
        private Network _discriminator;
        private Network _generator;

        /// <summary>
        /// Initializes a new instance of the <see cref="SymbolGAN"/>.
        /// </summary>
        /// <param name="shape">The shape of the training images <see cref="Tensor"/>s.</param>
        /// <param name="batchSize">The size of each training batch.</param>
        /// <param name="latentDimensions">The number of latent dimensions to be used when generating new images.</param>
        public SymbolGAN(TensorShape shape, int batchSize, int latentDimensions)
        {
            _imageLength = shape.Length;
            _imageWidth = shape.Width;
            _batchSize = batchSize;
            _latentDimensions = latentDimensions;
            _realInputs = new Tensor[_batchSize];
            _fakeInputs = new Tensor[_batchSize];
            _generatorInputs = new Tensor[ 2 * _batchSize];
            _realClassifications = new Vector(batchSize);
            _fakeClassifications = new Vector(batchSize);
            _generatorClassifications = new Vector(2 * batchSize);
            for (int i = 0; i < _batchSize; i++)
            {
                _realClassifications[i] = 1;
            }

            for (int i = 0; i < _batchSize; i++)
            {
                _fakeClassifications[i] = 0;
            }

            for (int i = 0; i < 2 * _batchSize; i++)
            {
                _generatorClassifications[i] = 1;
            }

            BuildGAN();
        }

        /// <summary>
        /// Generates a new batch of images.
        /// </summary>
        /// <returns>Returns an array of <see cref="Tensor"/>'s.</returns>
        public Tensor[] Test()
        {
            Tensor[] inputs = new Tensor[_batchSize];
            for (int j = 0; j < _batchSize; j++)
            {
                inputs[j] = TensorUtility.RandomTensor(_latentDimensions, 1, 1);
            }

            return _generator!.Generate(new List<Tensor[]> { inputs }, null, true);
        }

        /// <summary>
        /// Performs a single epoch using the given training data.
        /// </summary>
        public void Train(List<Tensor> trainingData)
        {
            for (int j = 0; j < trainingData.Count; j++)
            {
                int n = Utility.Random.Next(j, trainingData.Count);
                (trainingData[j], trainingData[n]) = (trainingData[n], trainingData[j]);
            }

            for (int i = 0; i < trainingData.Count / _batchSize; i++)
            {
                //Training the discriminator using real images.
                for (int j = 0; j < _batchSize; j++)
                {
                    _realInputs[j] = trainingData[i * _batchSize + j];
                }

                _discriminator.Train(new List<Tensor[]> { _realInputs }, null, _realClassifications);

                //Training the discriminator using fake images.
                for (int j = 0; j < _batchSize; j++)
                {
                    _fakeInputs[j] = TensorUtility.RandomTensor(_latentDimensions, 1, 1);
                }

                _generator.Generate(new List<Tensor[]> { _fakeInputs }, null, false);
                _discriminator.Train(null, _fakeClassifications);

                //Training the generator.
                for (int j = 0; j < 2 * _batchSize; j++)
                {
                    _generatorInputs[j] = TensorUtility.RandomTensor(_latentDimensions, 1, 1);
                }

                _generator.Train(new List<Tensor[]> { _generatorInputs }, null, _generatorClassifications);

                // ReSharper disable once PossibleLossOfFraction
                UpdateProgress(i / (float)(trainingData.Count / _batchSize));
            }
        }

        private static void UpdateProgress(float percent)
        {
            string output = "\rProgress: \t[";
            for (int i = 0; i < 10; i++)
            {
                if (percent > i / 10f)
                {
                    output += "█";
                }
                else if (percent > (2 * i - 1) / 20f)
                {
                    output += "▌";
                }
                else
                {
                    output += "-";
                }
            }

            output += "]";

            Console.Write(output);
        }

        private void BuildDiscriminator()
        {
            _discriminator = new Network(new CrossEntropyLoss(), new AdamHyperParameters()
            {
                LearningRate = 0.00002f,
                FirstMomentDecay = 0.5f
            });
            var init = new RandomNormal(0, 0.02f);
            _discriminator.AddInput(new TensorShape(_imageWidth, _imageLength, 1));
            _discriminator.AddActivation(Activation.Dropout);
            _discriminator.AddConvolution(64, 4, 2, init, activation: Activation.ReLU);
            _discriminator.AddConvolution(64, 4, 2, init, activation: Activation.ReLU);
            _discriminator.AddDense(1, GlorotNormal.Instance, activation: Activation.Sigmoid);

            _discriminator.Initialize();
        }

        private void BuildGAN()
        {
            BuildDiscriminator();
            BuildGenerator();

            _generator.StartUp(2 * _batchSize);
        }

        private void BuildGenerator()
        {
            _generator = new Network(_discriminator, new AdamHyperParameters()
            {
                LearningRate = 0.0002f,
                FirstMomentDecay = 0.5f
            });

            var init = new RandomNormal(0, 0.02f);
            _generator.AddInput(new TensorShape(_latentDimensions, 1, 1));

            int initialWidth = _imageWidth / 4;
            int initialLength = _imageLength / 4;

            _generator.AddDense(128 * initialWidth * initialLength, init, activation: Activation.ReLU);
            _generator.AddReshape(new TensorShape(initialWidth, initialLength, 128));
            _generator.AddTransConv(128, 4, 2, init, activation: Activation.ReLU);
            _generator.AddTransConv(128, 4, 2, init, activation: Activation.ReLU);
            _generator.AddConvolution(1, 7, initializer: init, activation: Activation.HyperbolicTangent);
        }
    }
}
