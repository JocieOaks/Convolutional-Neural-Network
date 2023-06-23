using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Networks
{
    /// <summary>
    /// The <see cref="Discriminator"/> class is a <see cref="Network"/> used to evaluate how closely an image matches it's label for a
    /// Least Square Conditional GAN.
    /// </summary>
    public class Discriminator : Network
    {
        private Vector[] _discriminatorGradients;
        private FeatureMap[,] _finalOutGradient;
        private Vector[] _generatorGradients;
        private Vector[] _imageVectors;
        private Vector[] _imageVectorsNorm;

        private Vector[] _previousImageGradient;

        [JsonProperty] private Vectorization _vectorizationLayer;

        /// <summary>
        /// Loads a <see cref="Discriminator"/> from a json file.
        /// </summary>
        /// <param name="file">The path of the json file.</param>
        /// <returns>Returns the deserialized <see cref="Discriminator"/>.</returns>
        public static Discriminator LoadFromFile(string file)
        {
            Discriminator discriminator = null;

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
                    discriminator = JsonConvert.DeserializeObject<Discriminator>(dataToLoad, new JsonSerializerSettings
                    {
                        TypeNameHandling = TypeNameHandling.Auto
                    });
                }
                catch (Exception e)
                {
                    Console.WriteLine("Error occured when trying to load data from file: " + file + "\n" + e.ToString());
                }
            }

            return discriminator;
        }

        /// <summary>
        /// Converts an images label into a <see cref="Vector"/>.
        /// </summary>
        /// <param name="bools">The bool portion of the label.</param>
        /// <param name="floats">The float portion of the label.</param>
        /// <returns>Returns a vector that represents an image's labels.</returns>
        public static Vector VectorizeLabel(bool[] bools, float[] floats)
        {
            Vector vector = new(bools.Length + floats.Length);
            for (int i = 0; i < bools.Length; i++)
            {
                vector[i] = bools[i] ? 1 : -1;
            }
            for (int i = 0; i < floats.Length; i++)
            {
                vector[bools.Length + i] = floats[i] * 2 - 1;
            }

            return vector.Normalized();
        }

        /// <summary>
        /// Backpropogates through the network, updating every <see cref="ILayer"/> in the <see cref="Discriminator"/>.
        /// </summary>
        /// <param name="gradients">The gradients for each vector output by the network.</param>
        /// <param name="input">The images and their associatd labels for this iteration of training.</param>
        /// <param name="learningRate">The learning rate defining the degree to which each layer should be updated.</param>
        public void Backwards(Vector[] gradients, ImageInput[] input, float learningRate)
        {
            FeatureMap[,] images = new FeatureMap[1, _batchSize];
            for (int i = 0; i < _batchSize; i++)
            {
                images[0, i] = input[i].Image;
            }

            _updateStep++;
            float correctionLearningRate = CorrectionLearningRate(learningRate, 0.9f, 0.999f);

            FeatureMap[,] transposedGradient = new FeatureMap[0, 0];
            Utility.StopWatch(() => _vectorizationLayer.Backwards(VectorNormalization.Backwards(_imageVectors, gradients), correctionLearningRate, 0.9f, 0.99f), $"Backwards {_vectorizationLayer.Name}", PRINTSTOPWATCH);

            FeatureMap[,] currentGradient = Utility.TransposeArray(transposedGradient);

            for (int j = Depth - 1; j > 0; j--)
            {
                Utility.StopWatch(() => _layers[j].Backwards(correctionLearningRate, 0.9f, 0.999f), $"Backwards {j} {_layers[j].Name}", PRINTSTOPWATCH);
            }

            //A learning rate of 0 indicates that the gradient is going to be used by a generator.
            if (correctionLearningRate == 0 || _layers[0] is not Convolution convolution)
            {
                Utility.StopWatch(() => _layers[0].Backwards(correctionLearningRate, 0.9f, 0.999f), $"Backwards {0} {_layers[0].Name}", PRINTSTOPWATCH);
            }
            else
            {
                Utility.StopWatch(() => convolution.BackwardsUpdateOnly(correctionLearningRate, 0.9f, 0.999f), $"Backwards {0} {_layers[0].Name}", PRINTSTOPWATCH);
            }
        }

        /// <summary>
        /// Forward propagates through the network to generate <see cref="Vector"/>s associated with each image.
        /// </summary>
        /// <param name="input">The images and their asscoiated labels.</param>
        /// <param name="inference">Determines whether the <see cref="Discriminator"/> is training or inferring. Defaults to false.</param>
        public void Forward(ImageInput[] input, bool inference = false)
        {
            for (int i = 0; i < _batchSize; i++)
            {
                _inputImages[0, i] = input[i].Image;
            }

            for (int j = 0; j < Depth; j++)
            {
                if (inference && _layers[j] is Dropout)
                {
                    Utility.StopWatch(() => (_layers[j] as Dropout).ForwardInference(), $"Forwards {j} {_layers[j].Name}", PRINTSTOPWATCH);
                }
                else
                {
                    Utility.StopWatch(() => _layers[j].Forward(), $"Forwards {j} {_layers[j].Name}", PRINTSTOPWATCH);
                }
            }

            //Normalization preferes featuremaps grouped by dimension first, while Vectorization prefers them to be grouped by batch member first.
            //This transposes the featuremaps to perform Vectorization.

            Utility.StopWatch(() => _imageVectors = _vectorizationLayer.Forward(), $"Forwards {_vectorizationLayer.Name}", PRINTSTOPWATCH);

            _imageVectorsNorm = VectorNormalization.Forward(_imageVectors);
        }

        /// <summary>
        /// Calculates the gradients for backpropagating through the <see cref="Generator"/>.
        /// </summary>
        /// <param name="inputs">The images and their associatd labels for this iteration of training.</param>
        /// <returns>Returns an array of <see cref="FeatureMap"/>s containing the <see cref="Generator"/> gradients.</returns>
        public FeatureMap[,] GeneratorGradient(ImageInput[] inputs)
        {
            Backwards(_generatorGradients, inputs, 0);
            FeatureMap[,] gradient = new FeatureMap[1, inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                gradient[0, i] = _finalOutGradient[0, i];
            }

            return gradient;
        }

        /// <inheritdoc/>
        public override void ResetNetwork()
        {
            base.ResetNetwork();

            _vectorizationLayer.Reset();
        }

        ///<inheritdoc/>
        public override void StartUp(int batchSize, int width, int length, int descriptionBools, int descriptionFloats)
        {
            base.StartUp(batchSize, width, length, descriptionBools, descriptionFloats);

            _finalOutGradient = new FeatureMap[1, batchSize];
            for (int j = 0; j < batchSize; j++)
            {
                _inputImages[0, j] = new FeatureMap(width, length);
            }

            FeatureMap[,] current = _inputImages;
            FeatureMap[,] gradients = _finalOutGradient;
            foreach (var layer in _layers)
            {
                (current, gradients) = layer.Startup(current, gradients);
            }

            _vectorizationLayer ??= new Vectorization();

            _vectorizationLayer.StartUp(Utility.TransposeArray(current), gradients, descriptionBools + descriptionFloats);

            _discriminatorGradients = new Vector[_batchSize];
            _generatorGradients = new Vector[_batchSize];

            _ready = true;
        }

        /// <summary>
        /// Performs one training iteration.
        /// The discriminator calculates the cosine similarity between the vector output of the network, and the images label.
        /// The loss is the square difference between the similarity and either 1 or -1 depending on whether the step is trying to maximize
        /// or minimize the similarity between the label and the vector.
        /// Note: Image labels should have at least two labels. If only one label is used, the cosine similarity between the image vector and the
        /// label vector will only have values of either 1 or -1 due to normalization. If only one label is desire, add a second label that has the
        /// same value for every image to avoid loss of information from normalization.
        /// The <see cref="Discriminator"/> can be used for a Non-Conditional GAN by using the same labels for every image.
        /// </summary>
        /// <param name="images">The images and their labels in the batch.</param>
        /// <param name="learningRate">The current learning rate for backpropagation.</param>
        /// <param name="momentum">The momentum for the gradients.</param>
        /// <param name="step">The current training step being performed.
        /// 0 - Training the <see cref="Discriminator"/> using real images.
        /// 1 - Training the <see cref="Discriminator"/> using fake images.
        /// 2 - Training the <see cref="Generator"/>.</param>
        /// <returns>Returns the loss for the current step.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="Discriminator"/> has not completed it's initial setup.</exception>
        public float Train(ImageInput[] images, float learningRate, float momentum, int step)
        {
            if (!_ready)
                throw new InvalidOperationException("Network has not finished setup");

            Forward(images);

            float totalLoss = 0;
            for (int i = 0; i < images.Length; i++)
            {
                Vector classificationVector = VectorizeLabel(images[i].Bools, images[i].Floats);
                float score = Vector.Dot(_imageVectorsNorm[i], classificationVector);
                float loss = step == 1 ? MathF.Pow(score + 1, 2) : MathF.Pow(score - 1, 2);
                totalLoss += loss;

                switch (step)
                {
                    case 0:
                        _discriminatorGradients[i] = loss * (2 * score - 2) * classificationVector;
                        break;

                    case 1:
                        _discriminatorGradients[i] = loss * (2 * score + 2) * classificationVector;
                        break;

                    case 2:
                        _generatorGradients[i] = loss * (2 * score - 2) * classificationVector;
                        break;
                }
            }

            if (step != 2)
            {
                if (_previousImageGradient != null)
                {
                    for (int i = 0; i < _batchSize; i++)
                    {
                        _discriminatorGradients[i] += _previousImageGradient[i] * momentum;
                    }
                }
                _previousImageGradient = _discriminatorGradients;

                Backwards(_discriminatorGradients, images, learningRate);
            }

            return totalLoss / images.Length;
        }
    }
}