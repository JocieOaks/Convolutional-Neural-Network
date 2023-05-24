// See https://aka.ms/new-console-template for more information

using Newtonsoft.Json;
using System;
using System.Buffers.Binary;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Reflection;
using System.Runtime.Serialization;

[Serializable]
public class CLIP
{
    public const float ASYMPTOTEERRORFACTOR = 1e-6f; //Used to avoid divide by zero or log of zero going to infinity.
    [JsonProperty]
    readonly private int _batchSize;

    [JsonProperty]
    readonly private InitialConvolutionLayer _initialConvolutionLayer;

    [JsonProperty]
    readonly private List<Layer> _layers = new();

    [JsonProperty]
    readonly private Transformer _transformer;

    readonly private int _vectorDimensions;
    [JsonProperty]
    readonly private VectorizationLayer _vectorizationLayer;

    Vector[] _descriptionVectors;
    private Vector[] _descriptionVectorsNorm;
    private FeatureMap[][][] _featureMaps;
    Vector[] _imageVectors;
    private Vector[] _imageVectorsNorm;
    //Used to avoid divide by zero or log of zero going to infinity.
    private FeatureMap[][] _initialFeatureMaps;

    private FeatureMap[][] _transposedFinalFeatureMap;
    public CLIP(int depth, int dimensions, int vectorDimensions, int batchSize, int descriptionBoolLength, int descriptionFloatLength, int width, int length)
    {
        FeatureMap[][] input = new FeatureMap[dimensions][];
        for (int i = 0; i < dimensions; i++)
        {
            input[i] = new FeatureMap[batchSize];
            for (int j = 0; j < batchSize; j++)
            {
                input[i][j] = new FeatureMap(width, length);
            }
        }

        _initialConvolutionLayer = new InitialConvolutionLayer(3, 1, ref input);
        _vectorizationLayer = new VectorizationLayer(vectorDimensions, dimensions);
        _transformer = new Transformer(descriptionBoolLength, descriptionFloatLength, vectorDimensions);
        _batchSize = batchSize;
        _vectorDimensions = vectorDimensions;

        _imageVectors = new Vector[batchSize];
        _descriptionVectors = new Vector[batchSize];
        _imageVectorsNorm = new Vector[batchSize];
        _descriptionVectorsNorm = new Vector[batchSize];
        _layers.Add(new BatchNormalizationLayer(ref input));
        for (int i = 0; i < depth; i++)
        {
            _layers.Add(new ConvolutionalLayer(3, 1, ref input));
            _layers.Add(new BatchNormalizationLayer(ref input));
            if (i < 6 && i % 2 == 0)
                _layers.Add(new AveragePoolLayer(2, ref input));
        }
        _featureMaps = new FeatureMap[Depth][][];
    }

    public CLIP() { }

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

    public static CLIP? LoadFromFile(string file)
    {
        CLIP? clip = null;

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
                clip = JsonConvert.DeserializeObject<CLIP>(dataToLoad, new JsonSerializerSettings
                {
                    TypeNameHandling = TypeNameHandling.Auto
                });
            }
            catch (Exception e)
            {
                Console.WriteLine("Error occured when trying to load data from file: " + file + "\n" + e.ToString());
            }
        }

        return clip;
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

    public void Backwards((Vector[] dL_dI, Vector[] dL_dD) gradients, (FeatureMap image, bool[] bools, float[] floats)[] input, float learningRate)
    {
        FeatureMap[] images = new FeatureMap[_batchSize];
        for (int i = 0; i < _batchSize; i++)
        {
            images[i] = input[i].image;

            _transformer.Backwards(input[i].bools, input[i].floats, VectorNormalizationLayer.Backwards(_descriptionVectors[i], gradients.dL_dD[i]), learningRate);
        }

        var watch = System.Diagnostics.Stopwatch.StartNew();
        FeatureMap[][] transposed = new FeatureMap[_batchSize][];
        for (int i = 0; i < _batchSize; i++)
        {
            transposed[i] = _vectorizationLayer.Backwards(_transposedFinalFeatureMap[i], VectorNormalizationLayer.Backwards(_imageVectors[i], gradients.dL_dI[i]), learningRate);
        }
        watch.Stop();
        var elapsedMs = watch.ElapsedMilliseconds;
        Console.WriteLine($"Backwards Vectorization Time: {elapsedMs / 1000f} s");

        FeatureMap[][] current = TransposeArray(transposed);

        for (int j = Depth - 1; j > 0; j--)
        {
            watch = System.Diagnostics.Stopwatch.StartNew();
            current = _layers[j].Backwards(_featureMaps[j - 1], current, learningRate);
            watch.Stop();
            elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine($"Backwards {j} Layer Time: {elapsedMs / 1000f} s");
        }
        watch = System.Diagnostics.Stopwatch.StartNew();
        current = _layers[0].Backwards(_initialFeatureMaps, current, learningRate);
        watch.Stop();
        elapsedMs = watch.ElapsedMilliseconds;
        Console.WriteLine($"Backwards 0 Layer Time: {elapsedMs / 1000f} s");
        watch = System.Diagnostics.Stopwatch.StartNew();
        _initialConvolutionLayer.Backwards(images, current, learningRate);
        watch.Stop();
        elapsedMs = watch.ElapsedMilliseconds;
        Console.WriteLine($"Backwards Initial Convolution Time: {elapsedMs / 1000f} s");
    }

    public void Forward((FeatureMap image, bool[] bools, float[] floats)[] input)
    {
        FeatureMap[][] current;
        FeatureMap[] images = new FeatureMap[_batchSize];
        for (int i = 0; i < _batchSize; i++)
        {
            images[i] = input[i].image;
            _descriptionVectors[i] = _transformer.Forward(input[i].bools, input[i].floats);
            _descriptionVectorsNorm[i] = VectorNormalizationLayer.Forward(_descriptionVectors[i]);
        }

        var watch = System.Diagnostics.Stopwatch.StartNew();
        current = _initialConvolutionLayer.Forward(images);
        watch.Stop();
        var elapsedMs = watch.ElapsedMilliseconds;
        Console.WriteLine($"Initial Convolution Time: {elapsedMs / 1000f} s");

        _initialFeatureMaps = current;

        for (int j = 0; j < Depth; j++)
        {
            watch = System.Diagnostics.Stopwatch.StartNew();
            current = _layers[j].Forward(current);
            _featureMaps[j] = current;
            watch.Stop();
            elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine($"{j} Layer Time: {elapsedMs / 1000f} s");
        }

        //Normalization preferes featuremaps grouped by dimension first, while Vectorization prefers them to be grouped by batch member first.
        //This transposes the featuremaps to perform Vectorization.

        _transposedFinalFeatureMap = TransposeArray(current);
        watch = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < _batchSize; i++)
        {
            _imageVectors[i] = _vectorizationLayer.Forward(_transposedFinalFeatureMap[i]);
            _imageVectorsNorm[i] = VectorNormalizationLayer.Forward(_imageVectors[i]);
        }
        watch.Stop();
        elapsedMs = watch.ElapsedMilliseconds;
        Console.WriteLine($"Vectorization Time: {elapsedMs / 1000f} s");
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

    [OnDeserialized]
    public void OnDeserialized(StreamingContext context)
    {
        _imageVectorsNorm = new Vector[_batchSize];
        _descriptionVectorsNorm = new Vector[_batchSize];
        _featureMaps = new FeatureMap[Depth][][];
    }

    public void SaveToFile(string file)
    {
        try
        {
            // create the directory the file will be written to if it doesn't already exist
            Directory.CreateDirectory(Path.GetDirectoryName(file)!);

            // serialize the C# game data object into Json
            string dataToStore = JsonConvert.SerializeObject(this, Formatting.Indented, new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.Auto
            });

            // write the serialized data to the file
            using (FileStream stream = File.Create(file))
            {
                using (StreamWriter writer = new(stream))
                {
                    writer.Write(dataToStore);
                }
            }

        }
        catch (System.Exception e)
        {
            Console.WriteLine("Error occured when trying to save data to file: " + file + "\n" + e.ToString());
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

    public (float, float) Test((FeatureMap image, bool[] bools, float[] floats)[] input)
    {
        Forward(input);
        float[,] matrix = Score();
        float loss = Loss(matrix);
        float accuracy = Accuracy(matrix);
        return (loss, accuracy);
    }

    public float Train((FeatureMap image, bool[] bools, float[] floats)[] input, float learningRate)
    {
        Forward(input);
        float[,] matrix = Score();
        float loss = Loss(matrix);
        (Vector[] imageGradient, Vector[] descriptionGradient) gradients = CalculateGradient(matrix, loss);

        Backwards(gradients, input, learningRate);
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
        T[,] transposed = new T[array.GetLength(0), array.GetLength(1)];
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