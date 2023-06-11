using Newtonsoft.Json;

public class Discriminator : ConvolutionalNeuralNetwork
{
    private Vector[] _generatorGradients;
    private Vector[] _discriminatorGradients;
    private Vector[] _imageVectors;
    private Vector[] _imageVectorsNorm;

    private Vector[] _previousImageGradient;

    protected FeatureMap[,] FinalOutGradient { get; set; }

    [JsonProperty] private VectorizationLayer _vectorizationLayer;

    public void Backwards(Vector[] gradients, ImageInput[] input, float learningRate)
    {
        FeatureMap[,] images = new FeatureMap[1, _batchSize];
        for (int i = 0; i < _batchSize; i++)
        {
            images[0, i] = input[i].Image;
        }

        FeatureMap[,] transposedGradient = new FeatureMap[0, 0];
        StopWatch(() => _vectorizationLayer.Backwards(VectorNormalizationLayer.Backwards(_imageVectors, gradients), learningRate), $"Backwards {_vectorizationLayer.Name}");

        FeatureMap[,] currentGradient = TransposeArray(transposedGradient);

        for (int j = Depth - 1; j > 0; j--)
        {
            StopWatch(() => _layers[j].Backwards(learningRate), $"Backwards {j} {_layers[j].Name}");
        }

        if(learningRate == 0)
        {
            StopWatch(() => _layers[0].Backwards(learningRate), $"Backwards {0} {_layers[0].Name}");
        }
        else
        {
            StopWatch(() => (_layers[0] as ConvolutionalLayer).BackwardsUpdateOnly(learningRate), $"Backwards {0} {_layers[0].Name}");
        }
    }

    public void Forward(ImageInput[] input, bool inference = false)
    {
        for (int i = 0; i < _batchSize; i++)
        {
            _inputImages[0, i] = input[i].Image;
        }

        for (int j = 0; j < Depth; j++)
        {
            if (inference && _layers[j] is DropoutLayer)
            {
                StopWatch(() => (_layers[j] as DropoutLayer).ForwardInference(), $"Forwards {j} {_layers[j].Name}");
            }
            else
            {
                StopWatch(() => _layers[j].Forward(), $"Forwards {j} {_layers[j].Name}");
            }
        }

        //Normalization preferes featuremaps grouped by dimension first, while Vectorization prefers them to be grouped by batch member first.
        //This transposes the featuremaps to perform Vectorization.

        StopWatch(() => _imageVectors = _vectorizationLayer.Forward(), $"Forwards {_vectorizationLayer.Name}");

        _imageVectorsNorm = VectorNormalizationLayer.Forward(_imageVectors);
    }

    public override void StartUp(int batchSize, int width, int length, int descriptionBools, int descriptionFloats)
    {
        base.StartUp(batchSize, width, length, descriptionBools, descriptionFloats);

        FinalOutGradient = new FeatureMap[1, batchSize];
        for (int j = 0; j < batchSize; j++)
        {
            _inputImages[0, j] = new FeatureMap(width, length);
        }

        FeatureMap[,] current = _inputImages;
        FeatureMap[,] gradients = FinalOutGradient;
        foreach (var layer in _layers)
        {
            (current, gradients) = layer.Startup(current, gradients);
        }

        _vectorizationLayer ??= new VectorizationLayer(descriptionBools + descriptionFloats);
        
        _vectorizationLayer.StartUp(TransposeArray(current), gradients);

        _discriminatorGradients = new Vector[_batchSize];
        _generatorGradients = new Vector[_batchSize];

        _ready = true;
    }

    public (FeatureMap[,], float, float) TrainDiscriminator(ImageInput[] real, ImageInput[] fake, float learningRate, float momentum)
    {
        if (!_ready)
            throw new InvalidOperationException("Network has not finished setup");

        ImageInput[] total = real.Concat(fake).ToArray();

        Forward(total);

        float discriminatorLoss = 0;
        for (int i = 0; i < real.Length; i++)
        {
            Vector classificationVector = ClassificationVectorization.Vectorize(real[i].Bools, real[i].Floats);
            float score = Vector.Dot(_imageVectorsNorm[i], classificationVector);
            float loss = MathF.Pow(score - 1, 2);
            discriminatorLoss += loss;
            _discriminatorGradients[i] = (2 * score - 2) * classificationVector;
            _generatorGradients[i] = new Vector(_classificationBoolsLength + _classificationFloatsLength);
        }

        float generatorLoss = 0;
        for (int i = 0; i < fake.Length; i++)
        {
            Vector classificationVector = ClassificationVectorization.Vectorize(fake[i].Bools, fake[i].Floats);
            float score = Vector.Dot(_imageVectorsNorm[i + real.Length], classificationVector);
            float loss = MathF.Pow(score + 1, 2);
            discriminatorLoss += loss;
            _discriminatorGradients[i + real.Length] = (2 * score + 2) * classificationVector;

            loss = MathF.Pow(score - 1, 2);
            generatorLoss += loss;
            _generatorGradients[i + real.Length] = (2 * score - 2) * classificationVector;
        }

        if (_previousImageGradient != null)
        {
            for (int i = 0; i < _batchSize; i++)
            {
                _discriminatorGradients[i] += _previousImageGradient[i] * momentum;
            }
        }
        _previousImageGradient = _discriminatorGradients;

        discriminatorLoss /= _batchSize;
        generatorLoss /= fake.Length;

        Backwards(_generatorGradients, total, 0);
        FeatureMap[,] gradient = new FeatureMap[1, fake.Length];
        for(int i = 0; i < fake.Length; i++)
        {
            gradient[0, i] = FinalOutGradient[0, i + real.Length];
        }

        Backwards(_discriminatorGradients, total, learningRate);

        return (gradient, discriminatorLoss, generatorLoss);
    }

    public float[,] Score(ImageInput[] input)
    {
        float[,] cosScores = new float[_batchSize, _batchSize];

        for (int i = 0; i < _batchSize; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                Vector classificationVector = ClassificationVectorization.Vectorize(input[i].Bools, input[i].Floats);
                cosScores[i, j] = Vector.Dot(_imageVectorsNorm[i], classificationVector);
            }
        }

        return cosScores;
    }

    public float[] ScoreIndividual(ImageInput[] input)
    {
        float[] scores = new float[_batchSize];
        for (int i = 0; i < _batchSize; i++)
        {
            Vector classificationVector = ClassificationVectorization.Vectorize(input[i].Bools, input[i].Floats);
            scores[i] = Vector.Dot(_imageVectorsNorm[i], classificationVector);
        }
        return scores;
    }

    public float Train(ImageInput[] input, float learningRate, float transformLearningRate, float momentum)
    {
        if (!_ready)
            throw new InvalidOperationException("Network has not finished setup");

        Forward(input);
        float[,] score = Score(input);
        float loss = Loss(score);
        _previousImageGradient = _discriminatorGradients;
        _discriminatorGradients = CalculateGradient(input, score, loss);
        if (_previousImageGradient != null)
        {
            for (int i = 0; i < _batchSize; i++)
            {
                _discriminatorGradients[i] += _previousImageGradient[i] * momentum;
            }
        }

        Backwards(_discriminatorGradients, input, learningRate);
        return loss;
    }

    private static Vector[] CalculateGradient(float[,] matrix, Vector[] gradientVectors, Vector[] dotVectors, float loss)
    {
        int length = gradientVectors.Length;
        Vector[] gradients = new Vector[length];
        for (int i = 0; i < length; i++)
        {
            gradients[i] = new Vector(gradientVectors[i].Length);
        }
        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < length; j++)
            {
                Vector[] nextGradients = i == j ?
                    DiagonalGradient(matrix, gradientVectors, dotVectors[i], loss, i) :
                    NonDiagonalGradient(matrix, gradientVectors, dotVectors[j], loss, i, j);
                for (int k = 0; k < length; k++)
                {
                    gradients[k] += nextGradients[k];
                }
            }
        }
        return gradients;
    }

    private static Vector[] DiagonalGradient(float[,] matrix, Vector[] gradientVectors, Vector dotVector, float loss, int index)
    {
        //Comments use TeX Comments for displaying mathematical formulae. Writing the full proof to maintain clarity.
        //tex:
        //$$f_{ij} = \textbf{x}^{(i)} \cdot \textbf{y}^{(j)}= x_1^{(i)}y_1^{(j)} + x_2^{(i)}y_2^{(j)} + \dotsb + x_n^{(i)}y_n^{(j)}$$
        //$$L_i = -\frac{1}{N}ln\left(\frac{e^{2f_{ii}-2}}{\sum\limits_je^{2f_{ji}-2}}\right) = -\frac{1}{N}\left(ln(e^{2f_{ii}-2}) - ln\left(\sum\limits_je^{2f_{ji}-2}\right)\right)$$
        //$$L_i = -\frac{1}{N}\left(2f_{ii} - 2 - ln\left(\sum\limits_je^{2f_{ji}-2}\right)\right) = -\frac{1}{N}(2f_{ii} - 2 - a_i)$$
        //$$a_i = ln\left(\sum\limits_je^{2f_{ji}-2}\right) = ln(b_i)$$
        //$$b_i = \sum\limits_je^{2f_{ji}-2}$$
        //tex:
        //$$\frac{d L_i}{d x_j^{(k)}}=-\frac{1}{N}\left(2\frac{d f_{ii}}{d x_j^{(k)}} - \frac{d a_i}{d x_j^{(k)}}\right)$$
        //$$\frac{d f_{ij}}{d x_k^{(l)}} = \delta_{il}y^{(j)}_k$$
        //$$\frac{d a_i}{d x_j^{(k)}} = \frac{1}{b_i}\frac{d b_i}{d x_j^{(k)}}$$
        //$$\frac{d b_i}{d x_j^{(k)}} = 2\sum\limits_le^{2f_{li}-2}\frac{d f_{li}}{d x_j^{(k)}} = 2\sum\limits_l\delta_{lk}e^{2f_{li}-2}y_j^{(i)}$$
        //$$\frac{d b_i}{d x_j^{(k)}} = 2e^{2f_ki-2}y_j^{(i)}$$
        //$$\frac{d L_i}{d x_j^{(k)}}=\frac{2e^{2f_ki-2}y_j^{(i)}}{N\sum\limits_je^{2f_{ji}-2}}-\frac{2\delta_{ik}y^{(i)}_j}{N}$$

        Vector[] gradients = new Vector[gradientVectors.Length];
        float b = 0;
        float invN = 1f / (matrix.GetLength(0) * matrix.GetLength(1));
        for (int i = 0; i < gradientVectors.Length; i++)
        {
            b += MathF.Exp(2 * matrix[i, index] - 2);
        }

        float mult = 2 * loss * invN / b;

        for (int i = 0; i < gradientVectors.Length; i++)
        {
            gradients[i] = mult * MathF.Exp(2 * matrix[i, index] - 2) * dotVector;
        }

        gradients[index] -= 2 * loss * invN * dotVector;

        return gradients;
    }

    private static Vector[] NonDiagonalGradient(float[,] matrix, Vector[] gradientVectors, Vector dotVector, float loss, int index1, int index2)
    {
        //Comments use TeX Comments for displaying mathematical formulae. Writing the full proof to maintain clarity.
        //tex:
        //$$f_{ij} = \textbf{x}^{(i)} \cdot \textbf{y}^{(j)}= x_1^{(i)}y_1^{(j)} + x_2^{(i)}y_2^{(j)} + \dotsb + x_n^{(i)}y_n^{(j)}$$
        //$$L_{ij} = -\frac{1}{N}ln\left(1 - \frac{e^{2f_{ij}-2}}{\sum\limits_ke^{2f_{kj}-2}} \right) = -\frac{ln(a_{ij})}{N}$$
        //$$a_{ij} = 1 - \frac{e^{2f_{ij}-2}}{\sum\limits_ke^{2f_{kj}-2}} = 1 - \frac{b_{ij}}{c_j}$$
        //$$b_{ij} = e^{2f_{ij} - 2}$$
        //$$c_i = \sum\limits_je^{2f_{ji}-2}$$
        //tex:
        //$$\frac{d L_{ij}}{d x_k^{(l)}} = -\frac{1}{Na_{ij}} \frac{da_{ij}}{d x_k^{(l)}}$$
        //$$\frac{d f_{ij}}{d x_k^{(l)}} = \delta_{il}y^{(j)}_k$$
        //$$\frac{da_{ij}}{d x_k^{(l)}} = \frac{\frac{db_{ij}}{d x_k^{(l)}} c_j - b \frac{dc_{j}}{d x_k^{(l)}}}{c_j^2}$$
        //$$\frac{db_{ij}}{d x_k^{(l)}} = 2e^{2f_{ij}-2}\frac{d f_{ij}}{d x_k^{(l)}} = 2\delta_{il}e^{2f_{ij}-2}y^{(j)}_k$$
        //$$\frac{dc_{i}}{d x_j^{(k)}} = 2\sum\limits_le^{2f_{li}-2}\frac{d f_{li}}{d x_j^{(k)}} = 2\sum\limits_l\delta_{lk}e^{2f_{li}-2}y_j^{(i)}$$
        //$$\frac{d c_i}{d x_j^{(k)}} = 2e^{2f_ki-2}y_j^{(i)}$$

        Vector[] gradients = new Vector[gradientVectors.Length];

        float b = MathF.Exp(matrix[index1, index2] - 2);
        float c = 0;

        for (int i = 0; i < gradientVectors.Length; i++)
        {
            c += MathF.Exp(2 * matrix[i, index2] - 2);
        }

        float a = 1 - b / c;
        float invc2 = MathF.Pow(c, -2);
        float mult = -loss / (a * matrix.GetLength(0) * matrix.GetLength(1));

        for (int i = 0; i < gradientVectors.Length; i++)
        {
            Vector cPrime = 2 * MathF.Exp(2 * matrix[i, index2] - 2) * dotVector;
            Vector aPrime = -b * cPrime;
            if (i == index1)
            {
                Vector bPrime = 2 * b * dotVector;
                aPrime += bPrime * c;
            }
            aPrime *= invc2;
            gradients[i] = mult * aPrime;
        }

        return gradients;
    }

    private Vector[] CalculateGradient(ImageInput[] input, float[,] matrix, float loss)
    {
        Vector[] classificationVectors = new Vector[_batchSize];
        for(int i = 0; i < _batchSize; i++)
        {
            classificationVectors[i] = ClassificationVectorization.Vectorize(input[i].Bools, input[i].Floats);
        }

        return CalculateGradient(matrix, _imageVectorsNorm, classificationVectors, loss);
    }

    public (float, float) Test(ImageInput[] input)
    {
        if (!_ready)
            throw new InvalidOperationException("Network has not finished setup");

        Forward(input, true);
        float[,] matrix = Score(input);
        float loss = Loss(matrix);
        float accuracy = Accuracy(matrix);
        return (loss, accuracy);
    }

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
}

