using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;
using System.Reflection.Emit;

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
        private Vector[] _finalOutput;
        private int _inputArea;

        delegate (float, Vector) LossFunction(ImageInput input, Vector vector, float targetValue);

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
        public void Backwards(Vector[] gradients, float learningRate)
        {
            if(learningRate > 0)
                _updateStep++;
            
            float correctionLearningRate = CorrectionLearningRate(learningRate, _firstMomentDecay, _secondMomentDecay);

            Vector[] sigmoidGradients = Sigmoid.Backward(_finalOutput, gradients);

            for (int i = 0; i < _batchSize; i++)
            {
                sigmoidGradients[i].CopyToBuffer(_endBuffers.InGradient.SubView(i * LabelCount, LabelCount));
            }

            for (int j = Depth - 1; j >= 0; j--)
            {
                Utility.StopWatch(() => _layers[j].Backwards(correctionLearningRate, _firstMomentDecay, _secondMomentDecay), $"Backwards {j} {_layers[j].Name}", PRINTSTOPWATCH);
            }
        }

        /// <summary>
        /// Forward propagates through the network to generate <see cref="Vector"/>s associated with each image.
        /// </summary>
        /// <param name="input">The images and their asscoiated labels.</param>
        /// <param name="inference">Determines whether the <see cref="Discriminator"/> is training or inferring. Defaults to false.</param>
        public void Forward(ImageInput[] input)
        {
            for (int i = 0; i < _batchSize; i++)
            {
                if (input[i].Image.Area != _inputArea)
                {
                    throw new ArgumentException("Input images are incorrectly sized.");
                }
                input[i].Image.CopyToBuffer(_startBuffers.Input.SubView(_inputArea * i, _inputArea));
            }

            for (int j = 0; j < Depth; j++)
            {
                Utility.StopWatch(() => _layers[j].Forward(), $"Forwards {j} {_layers[j].Name}", PRINTSTOPWATCH);
            }

            for(int i = 0; i < _batchSize; i++)
            {
                _finalOutput[i].SyncCPU(_endBuffers.Output.SubView(i * LabelCount, LabelCount));
            }

            _imageVectors = Sigmoid.Forward(_finalOutput);

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
        public FeatureMap[] GeneratorGradient()
        {
            Backwards(_generatorGradients, 0);
            for (int i = 0; i < _batchSize; i++)
            {
                _finalOutGradient[i].SyncCPU(_startBuffers.OutGradient.SubView(i * _inputArea, _inputArea));
            }

            return _finalOutGradient;
        }

        /// <inheritdoc/>
        public override void ResetNetwork()
        {
            base.ResetNetwork();
        }

        ///<inheritdoc/>
        public override void StartUp(int batchSize, int width, int length, int labelBools, int labelFloats, float learningRate, float firstDecay, float secondDecay)
        {
            base.StartUp(batchSize, width, length, labelBools, labelFloats, learningRate, firstDecay, secondDecay);

            _inputArea = width * length;

            if(_layers.Count == 0 || _layers.Last() is not FinalLayer)
            {
                _layers.Add(new Dense(labelFloats + labelBools));
            }

            Shape current = new Shape(width, length, 1);
            _finalOutGradient = new FeatureMap[batchSize];
            _finalOutput = new Vector[_batchSize];
            for (int j = 0; j < batchSize; j++)
            {
                _finalOutGradient[j] = new FeatureMap(width, length);
                _finalOutput[j] = new Vector(labelBools + labelFloats);
            }

            _startBuffers ??= new();
            _middleBuffers ??= new();

            IOBuffers inputBuffers = _startBuffers;
            IOBuffers outputBuffers = _middleBuffers;
            outputBuffers.OutputDimensionArea(width * length);

            foreach (var layer in _layers)
            {
                current = layer.Startup(current, inputBuffers, batchSize);
                if (layer is not IUnchangedLayer)
                {
                    (inputBuffers, outputBuffers) = (outputBuffers, inputBuffers);
                }
            }

            _endBuffers = outputBuffers;

            inputBuffers.Allocate(batchSize);
            outputBuffers.Allocate(batchSize);
            IOBuffers.SetCompliment(inputBuffers, outputBuffers);

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
        public float Train(ImageInput[] images, int step)
        {
            if (!_ready)
                throw new InvalidOperationException("Network has not finished setup");

            Forward(images);

            float totalLoss = 0;
            for (int i = 0; i < images.Length; i++)
            {
                float loss = 0;
                switch (step)
                {
                    case 0:
                        (loss, _discriminatorGradients[i]) = Loss(images[i], _imageVectorsNorm[i], 1);
                        break;

                    case 1:
                        (loss, _discriminatorGradients[i]) = Loss(images[i], _imageVectorsNorm[i], -1);
                        break;

                    case 2:
                        (loss, _generatorGradients[i]) = Loss(images[i], _imageVectorsNorm[i], 1);
                        break;
                }

                totalLoss += loss;
            }

            if (step != 2)
            { 
                Backwards(_discriminatorGradients, _learningRate);
            }

            return totalLoss / images.Length;
        }
        /// <summary>
        /// Caculates loss based on the log of the probability of the image being real or fake, where the probability is based on the angle 
        /// between the image vector and the classification vector.
        /// </summary>
        /// <param name="input">The <see cref="ImageInput"/> corresponding to <paramref name="vector"/>.</param>
        /// <param name="vector">The <see cref="Vector"/> produced by the <see cref="Discriminator"/>.</param>
        /// <param name="sign">Should be 1 or -1. -1 Inverts the probability for when the discriminator is testing the probability that a value is fake.</param>
        /// <returns>Returns the current loss, and the gradient <see cref="Vector"/>.</returns>
        private (float, Vector) CrossEntropyLoss(ImageInput input, Vector vector, float sign)
        {
            Vector classificationVector = VectorizeLabel(input.Bools, input.Floats);
            float loss;
            Vector gradient;
            float score = Vector.Dot(vector, classificationVector);
            if (sign == 1)
            {
                loss = -MathF.Log(score);
                gradient = (-1 / (score)) * classificationVector;
            }
            else
            {
                loss = -score * MathF.Log(1 - score);
                gradient = (1 / (1 - score)) * classificationVector;
            }
            return (loss, gradient);
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