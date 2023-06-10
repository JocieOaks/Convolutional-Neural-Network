using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;
using System.Drawing;
using System.IO;
using System.Reflection.Metadata.Ecma335;

[Serializable]
public partial class ConvolutionalNeuralNetwork
{
    //Used to avoid divide by zero or log of zero going to infinity.
    public const float ASYMPTOTEERRORFACTOR = 1e-6f; //Used to avoid divide by zero or log of zero going to infinity.

    private const bool PRINTSTOPWATCH = false;

    public static Context Context { get; } = Context.Create(builder => builder.Cuda());
    public static Accelerator Accelerator { get; } = Context.CreateCudaAccelerator(0);

    private Vector[] _descriptionGradient;
    private Vector[] _descriptionVectors;
    private Vector[] _descriptionVectorsNorm;
    private Vector[] _imageGradient;
    private Vector[] _imageVectors;
    private Vector[] _imageVectorsNorm;

    private Vector[] _previousDescriptionGradient;
    private Vector[] _previousImageGradient;

    [JsonConstructor]
    private ConvolutionalNeuralNetwork()
    {
    }

    public static Random Random { get; } = new Random();

    private int Depth => _layers.Count;

    public static float Accuracy(float[,] matrix)
    {
        int correct = 0;
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            int bestImageIndex = 0;
            float bestImageValue = matrix[i, 0];
            int bestDescriptionIndex = 0;
            float bestDescriptionValue = matrix[0, i];

            for (int j = 0; j < matrix.GetLength(1); j++)
            {
                if (matrix[i, j] > bestImageValue)
                {
                    bestImageIndex = j;
                    bestImageValue = matrix[i, j];
                }
                if (matrix[j, i] > bestDescriptionValue)
                {
                    bestDescriptionIndex = j;
                    bestDescriptionValue = matrix[j, i];
                }
            }
            if (bestImageIndex == i)
                correct++;
            if (bestDescriptionIndex == i)
                correct++;
        }
        return correct / (2f * matrix.GetLength(0));
    }

    public static float Loss(float[,] matrix)
    {
        float loss = 0.0f;
        int length = matrix.GetLength(0);
        for (int i = 0; i < length; i++)
        {
            float totalI = 0;
            float totalD = 0;
            for (int j = 0; j < length; j++)
            {
                totalI += MathF.Exp(2 * matrix[i, j] - 2);
                totalD += MathF.Exp(2 * matrix[j, i] - 2);
            }

            for (int j = 0; j < length; j++)
            {
                if (i == j)
                    loss += MathF.Log(MathF.Exp(2 * matrix[i, j] - 2) * MathF.Exp(2 * matrix[j, i] - 2) / totalD / totalI);
                else
                    loss += MathF.Log((totalD - MathF.Exp(2 * matrix[j, i] - 2)) / totalD) + MathF.Log((totalI - MathF.Exp(2 * matrix[i, j] - 2)) / totalI);
            }
        }
        return -loss / (length * length);
    }

    public static float RandomGauss(float mean, float stdDev)
    {
        float u1 = 1 - (float)Random.NextDouble(); //uniform(0,1] random doubles
        float u2 = 1 - (float)Random.NextDouble();
        float randStdNormal = MathF.Sqrt(-2 * MathF.Log(u1)) * MathF.Sin(2 * MathF.PI * u2); //random normal(0,1)
        return mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)
    }

    public void Backwards((Vector[] image, Vector[] description) gradients, ImageInput[] input, float learningRate, float transformLearningRate)
    {
        FeatureMap[,] images = new FeatureMap[1, _batchSize];
        for (int i = 0; i < _batchSize; i++)
        {
            images[0, i] = input[i].Image;

            _transformer.Backwards(input[i].Bools, input[i].Floats, VectorNormalizationLayer.Backwards(_descriptionVectors[i], gradients.description[i]), transformLearningRate);
        }

        FeatureMap[,] transposedGradient = new FeatureMap[0, 0];
        StopWatch(() => _vectorizationLayer.Backwards(VectorNormalizationLayer.Backwards(_imageVectors, gradients.image), learningRate), $"Backwards {_vectorizationLayer.Name}");

        FeatureMap[,] currentGradient = TransposeArray(transposedGradient);

        for (int j = Depth - 1; j > 0; j--)
        {
            StopWatch(() => _layers[j].Backwards(learningRate), $"Backwards {j} {_layers[j].Name}");
        }
        StopWatch(() => (_layers[0] as ConvolutionalLayer).BackwardsFilterOnly(learningRate), $"Backwards {0} {_layers[0].Name}");
    }

    public void Forward(ImageInput[] input, bool inference = false)
    {
        for (int i = 0; i < _batchSize; i++)
        {
            _inputImages[0, i] = input[i].Image;
            _descriptionVectors[i] = _transformer.Forward(input[i].Bools, input[i].Floats);
            _descriptionVectorsNorm[i] = VectorNormalizationLayer.Forward(_descriptionVectors[i]);
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

    public FeatureMap[,] ForwardGenerate(ImageInput[] input, bool inference = false)
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

        return _layers.Last().Outputs;
    }

    public void BackWardsGenerate(FeatureMap gradients, ImageInput[] input, float learningRate)
    {
        FeatureMap[,] images = new FeatureMap[1, _batchSize];
        for (int i = 0; i < _batchSize; i++)
        {
            images[0, i] = input[i].Image;
        }

        _firstInGradient[0,0] = gradients;

        for (int j = Depth - 1; j > 0; j--)
        {
            StopWatch(() => _layers[j].Backwards(learningRate), $"Backwards {j} {_layers[j].Name}");
        }
        StopWatch(() => (_layers[0] as ConvolutionalLayer).BackwardsFilterOnly(learningRate), $"Backwards {0} {_layers[0].Name}");
    }


    public IEnumerable<(float, float)> GradientTest(int vectorCount, int vectorLength)
    {
        _imageVectorsNorm = new Vector[vectorCount];
        _descriptionVectorsNorm = new Vector[vectorCount];
        for (int i = 0; i < vectorCount; i++)
        {
            Vector newImageVector = new Vector(vectorLength);
            Vector newDescriptionVector = new Vector(vectorLength);
            for (int j = 0; j < vectorLength; j++)
            {
                newImageVector[j] = (float)(Random.NextDouble() * 2 - 1);
                newDescriptionVector[j] = (float)(Random.NextDouble() * 2 - 1);
            }
            _imageVectorsNorm[i] = newImageVector.Normalized();
            _descriptionVectorsNorm[i] = newDescriptionVector.Normalized();
        }

        float[,] matrix = Score();
        float loss = Loss(matrix);
        float accuracy = Accuracy(matrix);
        yield return (loss, accuracy);
        for (int i = 0; i < 10; i++)
        {
            (Vector[] imageGradients, Vector[] descriptionGradients) = CalculateGradient(matrix, loss);
            for (int j = 0; j < vectorCount; j++)
            {
                _imageVectorsNorm[j] -= imageGradients[j] * 2;
                _descriptionVectorsNorm[j] -= descriptionGradients[j] * 2;
            }
            matrix = Score();
            loss = Loss(matrix);
            accuracy = Accuracy(matrix);
            yield return (loss, accuracy);
        }
    }

    public (FeatureMap, float) Discriminate(ImageInput input)
    {
        if (!_ready)
            throw new InvalidOperationException("Network has not finished setup");

        ImageInput[] augmented = new ImageInput[_batchSize];
        augmented[0] = input;
        for (int i = 1; i < _batchSize; i++)
        {
            double random = Random.NextDouble();
            augmented[i] = new ImageInput
            {
                Image = random switch
                {
                    < .1 => Augmentations.HorizontalFlip(input.Image),
                    < .4 => Augmentations.GaussianNoise(input.Image),
                    < .7 => Augmentations.RandomBrightness(input.Image),
                    _ => Augmentations.RandomSaturation(input.Image)
                },
                Bools = input.Bools,
                Floats = input.Floats
            };

            using (Context context = Context.Create(builder => builder.Cuda()))
            {
                Accelerator accelerator = ConvolutionalNeuralNetwork.Accelerator;
                Bitmap generatedBitmap = augmented[i].Image.ConstructBitmap(accelerator, true);
            }
        }
        Forward(augmented);

        _imageGradient ??= new Vector[_batchSize];

        float totalLoss = 0;
        for (int i = 0; i < _batchSize; i++)
        {
            float score = Vector.Dot(_imageVectorsNorm[i], _descriptionVectorsNorm[i]);
            float loss = -(score - 1);
            if(float.IsNaN(loss))
            {
                return (null, 2);
            }
            totalLoss += loss;
            _imageGradient[i] =  loss * _descriptionVectorsNorm[i];
        }
        totalLoss /= _batchSize;
        Console.WriteLine($"Loss {totalLoss}");

        BackGradient(_imageGradient, augmented);
        FeatureMap gradient = new FeatureMap(input.Image.Width, input.Image.Length);

        for (int j = 0; j < gradient.Length; j++)
        {
            for (int k = 0; k < gradient.Width; k++)
            {
                for (int i = 0; i < _batchSize; i++)
                {
                    gradient[k, j] += _finalOutGradient[0, i][k, j];
                }
                gradient[k, j] /= _batchSize;
            }
        }
        return (gradient, totalLoss);
    }

    public void BackGradient(Vector[] gradient, ImageInput[] input)
    {
        FeatureMap[,] images = new FeatureMap[1, _batchSize];
        for (int i = 0; i < _batchSize; i++)
        {
            images[0, i] = input[i].Image;
        }

        FeatureMap[,] transposedGradient = new FeatureMap[0, 0];
        StopWatch(() => _vectorizationLayer.Backwards(VectorNormalizationLayer.Backwards(_imageVectors, gradient), 0), $"Backwards {_vectorizationLayer.Name}");

        FeatureMap[,] currentGradient = TransposeArray(transposedGradient);

        for (int j = Depth - 1; j >= 0; j--)
        {
            StopWatch(() => _layers[j].BackwardsNoUpdate(), $"Backwards {j} {_layers[j].Name}");
        }
    }

    public void Initialize(ImageInput[] input)
    {
        Forward(input);
        bool changed;
        do
        {
            changed = false;
            for (int j = 0; j < _batchSize; j++)
            {
                float dot = Vector.Dot(_imageVectorsNorm[j], _descriptionVectorsNorm[j]);
                if (dot > 0.3 || dot < -0.1)
                {
                    Vector gradient = dot * _imageVectorsNorm[j];
                    changed = true;

                    _transformer.Backwards(input[j].Bools, input[j].Floats, VectorNormalizationLayer.Backwards(_descriptionVectors[j], gradient), 0.001f);
                    _descriptionVectors[j] = _transformer.Forward(input[j].Bools, input[j].Floats);
                    _descriptionVectorsNorm[j] = VectorNormalizationLayer.Forward(_descriptionVectors[j]);
                }
            }
        } while (changed);
    }

    public void PrintFeatureMaps(string directory, string name, int batchIndex)
    {
        directory = Path.Combine(directory, name);
        try
        {
            // create the directory the file will be written to if it doesn't already exist
            Directory.CreateDirectory(directory);
        }
        catch (System.Exception e)
        {
            Console.WriteLine("Error occured when trying to create director: " + directory + "\n" + e.ToString());
        }
Context context = ConvolutionalNeuralNetwork.Context;
        Accelerator accelerator = ConvolutionalNeuralNetwork.Accelerator;
        string layerDirectory;
        for (int i = 0; i < Depth; i++)
        {
            if (_layers[i] is BatchNormalizationLayer)
            {
                layerDirectory = Path.Combine(directory, $"{i} {_layers[i].Name}");
                Directory.CreateDirectory(layerDirectory);
                for (int j = 0; j < _layers[i].OutputDimensions; j++)
                {
                    PrintFeatureMap(_layers[i].Outputs[j, batchIndex], Path.Combine(layerDirectory, $"{name} {j}.png"), accelerator);
                }
            }
        }
    }

    private void PrintFeatureMap(FeatureMap map, string file, Accelerator accelerator)
    {
        Bitmap image = map.ConstructBitmap(accelerator);
        try
        {
            image.Save(file, System.Drawing.Imaging.ImageFormat.Png);
        }
        catch (System.Exception e)
        {
            Console.WriteLine("Error occured when trying to save image: " + file + "\n" + e.ToString());
        }
    }

    public float[,] Score()
    {
        float[,] cosScores = new float[_batchSize, _batchSize];

        for (int i = 0; i < _batchSize; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                cosScores[i, j] = Vector.Dot(_imageVectorsNorm[i], _descriptionVectorsNorm[j]);
            }
        }

        return cosScores;
    }

    public float[] ScoreIndividual()
    {
        float[] scores = new float[_batchSize];
        for (int i = 0; i < _batchSize; i++)
        {
            scores[i] = Vector.Dot(_imageVectorsNorm[i], _descriptionVectorsNorm[i]);
        }
        return scores;
    }

    public (float, float) Test(ImageInput[] input)
    {
        if (!_ready)
            throw new InvalidOperationException("Network has not finished setup");

        Forward(input, true);
        float[,] matrix = Score();
        float loss = Loss(matrix);
        float accuracy = Accuracy(matrix);
        return (loss, accuracy);
    }

    public float Train(ImageInput[] input, float learningRate, float transformLearningRate, float momentum)
    {
        if (!_ready)
            throw new InvalidOperationException("Network has not finished setup");

        Forward(input);
        float[,] score = Score();
        float loss = Loss(score);
        _previousDescriptionGradient = _descriptionGradient;
        _previousImageGradient = _imageGradient;
        (_imageGradient, _descriptionGradient) = CalculateGradient(score, loss);
        if (_previousImageGradient != null)
        {
            for (int i = 0; i < _batchSize; i++)
            {
                _descriptionGradient[i] += _previousDescriptionGradient[i] * momentum;
                _imageGradient[i] += _previousImageGradient[i] * momentum;
            }
        }

        Backwards((_imageGradient, _descriptionGradient), input, learningRate, transformLearningRate);
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

    private static void StopWatch(Action func, string processName)
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        func();
        watch.Stop();
        var elapsedMs = watch.ElapsedMilliseconds;
        if (PRINTSTOPWATCH)
            Console.WriteLine($"Time: {elapsedMs / 1000f:F3} s {processName}");
    }

    private static T[][] TransposeArray<T>(T[][] array)
    {
        T[][] transposed = new T[array[0].Length][];
        for (int i = 0; i < transposed.Length; i++)
        {
            transposed[i] = new T[array.Length];
            for (int j = 0; j < transposed[i].Length; j++)
            {
                transposed[i][j] = array[j][i];
            }
        }
        return transposed;
    }

    private static T[,] TransposeArray<T>(T[,] array)
    {
        T[,] transposed = new T[array.GetLength(1), array.GetLength(0)];
        for (int i = 0; i < transposed.GetLength(0); i++)
        {
            for (int j = 0; j < transposed.GetLength(1); j++)
            {
                transposed[i, j] = array[j, i];
            }
        }
        return transposed;
    }

    private (Vector[], Vector[]) CalculateGradient(float[,] matrix, float loss)
    {
        return (CalculateGradient(matrix, _imageVectorsNorm, _descriptionVectorsNorm, loss),
            CalculateGradient(TransposeArray(matrix), _descriptionVectorsNorm, _imageVectorsNorm, loss));
    }
}