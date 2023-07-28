using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers;
using ConvolutionalNeuralNetwork.Layers.Initializers;
using ConvolutionalNeuralNetwork.Layers.Activations;
using Newtonsoft.Json;
using ConvolutionalNeuralNetwork.Layers.Weighted;

namespace ConvolutionalNeuralNetwork.Networks
{
    /// <summary>
    /// The <see cref="Discriminator"/> class is a <see cref="Network"/> used to evaluate how closely an image matches it's label for a
    /// Least Square Conditional GAN.
    /// </summary>
    public class Discriminator : Network
    {
        private Vector[] _discriminatorGradients;
        private FeatureMap[] _finalOutGradient;
        private Vector[] _generatorGradients;
        private Vector[] _imageVectors;
        private Vector[] _imageVectorsNorm;
        private int _inputArea;

        private delegate (float, bool, Vector) LossFunction(ImageInput input, Vector vector, float targetValue);

        /// <value>The function to use to calculate loss.</value>
        private LossFunction Loss => CrossEntropyLoss;

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
                    using (StreamReader r = new(file))
                    {
                        using (JsonReader reader = new JsonTextReader(r))
                        {
                            JsonSerializer serializer = new();
                            serializer.TypeNameHandling = TypeNameHandling.Auto;
                            discriminator = serializer.Deserialize<Discriminator>(reader);
                        }
                    }
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
        public void Backwards(Vector[] gradients, int batchSize, bool update = true)
        {
            _adamHyperParameters.Update(update);

            for (int i = 0; i < batchSize; i++)
            {
                gradients[i].CopyToBuffer(InGradient.SubView(i * LabelCount, LabelCount));
            }

            for (int j = Depth - 1; j >= 0; j--)
            {
                Utility.StopWatch(() => _layers[j].Backwards(batchSize), $"Backwards {j} {_layers[j].Name}", PRINTSTOPWATCH);
            }
        }

        /// <summary>
        /// Forward propagates through the network to generate <see cref="Vector"/>s associated with each image.
        /// </summary>
        /// <param name="input">The images and their asscoiated labels.</param>
        /// <param name="inference">Determines whether the <see cref="Discriminator"/> is training or inferring. Defaults to false.</param>
        public void Forward(ImageInput[] input)
        {
            int batchSize = input.Length;

            for (int j = 0; j < Depth; j++)
            {
                Utility.StopWatch(() => _layers[j].Forward(batchSize), $"Forwards {j} {_layers[j].Name}", PRINTSTOPWATCH);
            }

            for (int i = 0; i < batchSize; i++)
            {
                _imageVectors[i].SyncCPU(Output.SubView(i * LabelCount, LabelCount));
            }

            if (_floatLabels + _boolLabels > 1)
                _imageVectorsNorm = VectorNormalization.Forward(_imageVectors);
            else
                _imageVectorsNorm = _imageVectors;
        }

        /// <summary>
        /// Calculates the gradients for backpropagating through the <see cref="Generator"/>.
        /// </summary>
        /// <param name="inputs">The images and their associatd labels for this iteration of training.</param>
        /// <returns>Returns an array of <see cref="FeatureMap"/>s containing the <see cref="Generator"/> gradients.</returns>
        public void GeneratorGradient(int batchSize)
        {
            Backwards(_generatorGradients, batchSize, false);
        }

        /// <inheritdoc/>
        public override void ResetNetwork()
        {
            base.ResetNetwork();
        }

        ///<inheritdoc/>
        public override void StartUp(int maxBatchSize, int width, int length, int labelBools, int labelFloats, AdamHyperParameters hyperParameters, int inputChannels)
        {
            base.StartUp(maxBatchSize, width, length, labelBools, labelFloats, hyperParameters, inputChannels);

            _inputArea = width * length;

            Shape current = new(width, length, inputChannels);
            _finalOutGradient = new FeatureMap[maxBatchSize];
            _imageVectors = new Vector[maxBatchSize];
            for (int j = 0; j < maxBatchSize; j++)
            {
                _finalOutGradient[j] = new FeatureMap(current);
                _imageVectors[j] = new Vector(LabelCount);
            }

            InitializeLayers(ref current, maxBatchSize);

            if (current.Area != LabelCount)
            {
                var dense = new Dense(LabelCount, GlorotUniform.Instance);
                dense.SetUpWeights(_adamHyperParameters);
                _layers.Add(dense);
                _layers.Add(new HyperTan());
                InitializeLayers(ref current, maxBatchSize);
            }

            _discriminatorGradients = new Vector[maxBatchSize];
            _generatorGradients = new Vector[maxBatchSize];

            _ready = true;
        }

        private void LoadImages(ImageInput[] input)
        {
            int batchSize = input.Length;

            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < _inputChannels; j++)
                {
                    if (input[i].Image[j].Area != _inputArea)
                    {
                        throw new ArgumentException("Input images are incorrectly sized.");
                    }
                    input[i].Image[j].CopyToBuffer(_startBuffers.Input.SubView(_inputArea * (i * _inputChannels + j), _inputArea));
                }
            }
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
        public (float, float) Train(ImageInput[] images, int step)
        {
            if (!_ready)
                throw new InvalidOperationException("Network has not finished setup");

            //if (step == 0)
            {
                LoadImages(images);
            }

            Forward(images);

            float totalLoss = 0;
            float hitCount = 0;
            for (int i = 0; i < images.Length; i++)
            {
                float loss = 0;
                bool accurate = false;
                switch (step)
                {
                    case 0:
                        (loss, accurate,  _discriminatorGradients[i]) = Loss(images[i], _imageVectorsNorm[i], 1);
                        break;

                    case 1:
                        (loss, accurate, _discriminatorGradients[i]) = Loss(images[i], _imageVectorsNorm[i], -1);
                        break;

                    case 2:
                        (loss, accurate, _generatorGradients[i]) = Loss(images[i], _imageVectorsNorm[i], 1);
                        break;
                }

                if (accurate)
                    hitCount++;
                
                totalLoss += loss;
            }

            if (step != 2)
            {
                Backwards(_discriminatorGradients, images.Length);
            }

            return (totalLoss / images.Length, hitCount / images.Length);
        }

        /// <summary>
        /// Caculates loss based on the log of the probability of the image being real or fake, where the probability is based on the angle
        /// between the image vector and the classification vector.
        /// </summary>
        /// <param name="input">The <see cref="ImageInput"/> corresponding to <paramref name="vector"/>.</param>
        /// <param name="vector">The <see cref="Vector"/> produced by the <see cref="Discriminator"/>.</param>
        /// <param name="sign">Should be 1 or -1. -1 Inverts the probability for when the discriminator is testing the probability that a value is fake.</param>
        /// <returns>Returns the current loss, and the gradient <see cref="Vector"/>.</returns>
        private (float, bool, Vector) CrossEntropyLoss(ImageInput input, Vector vector, float sign)
        {
            Vector classificationVector = VectorizeLabel(input.Bools, input.Floats);
            float loss;
            Vector gradient;
            float score = (Vector.Dot(vector, classificationVector) + 1) / 2;
            bool accurate;
            if (sign == 1)
            {
                loss = -MathF.Log(score + Utility.ASYMPTOTEERRORCORRECTION);
                gradient = (-1 / (score + Utility.ASYMPTOTEERRORCORRECTION)) * classificationVector;
                accurate = score > 0.5;
            }
            else
            {
                loss = -MathF.Log(1 - score + Utility.ASYMPTOTEERRORCORRECTION);
                gradient = (1 / (1 - score + Utility.ASYMPTOTEERRORCORRECTION)) * classificationVector;
                accurate = score < 0.5;
            }

            return (loss, accurate, gradient);
        }

        /// <summary>
        /// Caculates loss based on the square difference between the cosine similarity of the image vector and it's classification vector, and a give target value.
        /// </summary>
        /// <param name="input">The <see cref="ImageInput"/> corresponding to <paramref name="vector"/>.</param>
        /// <param name="vector">The <see cref="Vector"/> produced by the <see cref="Discriminator"/>.</param>
        /// <param name="targetValue">The target value that the cosine similarity should be equal to.</param>
        /// <returns>Returns the current loss, and the gradient <see cref="Vector"/>.</returns>
        private (float, Vector) LeastSquareLoss(ImageInput input, Vector vector, float targetValue)
        {
            Vector classificationVector = VectorizeLabel(input.Bools, input.Floats);
            float score = Vector.Dot(vector, classificationVector);
            float loss = MathF.Pow(score - targetValue, 2);

            return (loss, loss * 2 * (score - targetValue) * classificationVector);
        }
    }
}