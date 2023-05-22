// See https://aka.ms/new-console-template for more information

using Newtonsoft.Json;
using System;
using System.Buffers.Binary;
using System.ComponentModel.DataAnnotations;
using System.Runtime.Serialization;

[Serializable]
public class CLIP
{
    public static Random Random { get; } = new Random();
    public const float ASYMPTOTEERRORFACTOR = 1e-6f; //Used to avoid divide by zero or log of zero going to infinity.

    [JsonProperty]
    readonly private VectorizationLayer _vectorizationLayer;
    [JsonProperty]
    readonly private Transformer _transformer;
    [JsonProperty]
    readonly private InitialConvolutionLayer _initialConvolutionLayer;
    [JsonProperty]
    readonly private List<Layer> _layers = new();

    private FeatureMap[][] _initialFeatureMaps;
    private FeatureMap[][][] _featureMaps;
    private FeatureMap[][] _transposedFinalFeatureMap;
    private Vector[] _imageVectors;
    private Vector[] _descriptionVectors;

    private int Depth => _layers.Count;
    [JsonProperty]
    readonly private int _batchSize;
    readonly private int _vectorDimensions;

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

        _initialConvolutionLayer = new InitialConvolutionLayer(dimensions, 3, 2, ref input);
        _vectorizationLayer = new VectorizationLayer(vectorDimensions, dimensions);
        _transformer = new Transformer(descriptionBoolLength, descriptionFloatLength, vectorDimensions);
        _batchSize = batchSize;
        _vectorDimensions = vectorDimensions;

        _imageVectors = new Vector[batchSize];
        _descriptionVectors = new Vector[batchSize];

        for (int i = 0; i < depth; i++)
        {
            _layers.Add(new ConvolutionalLayer(dimensions, 3, 1, ref input));
            _layers.Add(new NormalizationLayer(dimensions, ref input));
            if (i < 6 && i % 2 == 0)
                _layers.Add(new AveragePoolLayer(dimensions, 2, ref input));
        }
        _featureMaps = new FeatureMap[Depth][][];
    }

    public CLIP() { }

    [OnDeserialized]
    public void OnDeserialized(StreamingContext context)
    {
        _imageVectors = new Vector[_batchSize];
        _descriptionVectors = new Vector[_batchSize];
        _featureMaps = new FeatureMap[Depth][][];
    }

    public void Forward((FeatureMap image, bool[] bools, float[] floats)[] input)
    {
        FeatureMap[][] current;
        FeatureMap[] images = new FeatureMap[_batchSize];
        for (int i = 0; i < _batchSize; i++)
        {
            images[i] = input[i].image;
            _descriptionVectors[i] = _transformer.Forward(input[i].bools, input[i].floats).Normalized();
        }

        current = _initialConvolutionLayer.Forward(images);
        _initialFeatureMaps = current;

        for (int j = 0; j < Depth; j++)
        {
            current = _layers[j].Forward(current);
            _featureMaps[j] = current;
        }

        //Normalization preferes featuremaps grouped by dimension first, while Vectorization prefers them to be grouped by batch member first.
        //This transposes the featuremaps to perform Vectorization.

        _transposedFinalFeatureMap = TransposeArray(current);

        for (int i = 0; i < _batchSize; i++)
        {
            _imageVectors[i] = _vectorizationLayer.Forward(_transposedFinalFeatureMap[i]);
        }
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

    public float[,] Score()
    {
        float[,] cosScores = new float[_batchSize, _batchSize];

        for (int i = 0; i < _batchSize; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                cosScores[i, j] = Vector.Dot(_imageVectors[i], _descriptionVectors[j]);
            }
        }

        return cosScores;
    }

    /// <summary>
    /// One way to minimize <see cref="Loss"/> is by making every image vector have a negative dot product with every description vector,
    /// despite the intention being the dot product is negative for every pair except the expected pair, which has a dot product of 1.
    /// Once it's reached this state, the gradient becomes zero, and the model can no longer learn. In order to encourage differentiation
    /// between the resulting vectors from deep learning, we can add to the loss function based on the dot products between the vectors themselves.
    /// </summary>
    /// <returns></returns>
    public float[,] VectorColinearityScore()
    {
        float[,] imageDots = new float[_batchSize, _batchSize];
        for (int i = 0; i < _batchSize; i++)
        {
            for (int j = i + 1; j < _batchSize; j++)
            {
                imageDots[i, j] = imageDots[j, i] = MathF.Exp(10 * Vector.Dot(_imageVectors[i], _imageVectors[j]) - 5.5f);
            }
        }
        return imageDots;
    }

    public static float VectorColinearityLoss(float[,] matrix)
    {
        float loss = 0;
        int length = matrix.GetLength(0);
        for (int i = 0; i < length; i++)
        {
            for (int j = i + 1; j < length; j++)
            {
                loss += matrix[i, j];
            }
        }

        return loss / (length * length);
    }

    public Vector[] VectorColinearityGradient(float[,] matrix, float loss)
    {
        float mult = 5 * loss / (_batchSize * _batchSize);
        Vector[] gradients = new Vector[_batchSize];
        for (int i = 0; i < _batchSize; i++)
        {
            gradients[i] = new Vector(_imageVectors[i].Length);
            for (int j = 0; j < _batchSize; j++)
            {
                if (i != j)
                {
                    gradients[i] += mult * matrix[i, j] * _imageVectors[j];
                }
            }
        }
        return gradients;
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

    public void Backwards((Vector[] dL_dI, Vector[] dL_dD) gradients, (FeatureMap image, bool[] bools, float[] floats)[] input, float learningRate)
    {
        FeatureMap[] images = new FeatureMap[_batchSize];
        for (int i = 0; i < _batchSize; i++)
        {
            images[i] = input[i].image;
            _transformer.Backwards(input[i].bools, input[i].floats, gradients.dL_dD[i], learningRate);
        }

        FeatureMap[][] transposed = new FeatureMap[_batchSize][];
        for (int i = 0; i < _batchSize; i++)
        {
            transposed[i] = _vectorizationLayer.Backwards(_transposedFinalFeatureMap[i], gradients.dL_dI[i], learningRate);
        }

        FeatureMap[][] current = TransposeArray(transposed);

        for (int j = Depth - 1; j > 0; j--)
        {
            current = _layers[j].Backwards(_featureMaps[j - 1], current, learningRate);
        }
        current = _layers[0].Backwards(_initialFeatureMaps, current, learningRate);
        _initialConvolutionLayer.Backwards(images, current, learningRate);
    }

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
        /*float[,] colinearMatrix = VectorColinearityScore();
        float colinearLoss = VectorColinearityLoss(colinearMatrix);
        Vector[] colinearGradient = VectorColinearityGradient(colinearMatrix, colinearLoss);

        for (int i = 0; i < colinearGradient.Length; i++)
        {
            gradients.imageGradient[i] += 0.25f * colinearGradient[i];
        }*/

        Backwards(gradients, input, learningRate);
        return loss;// + colinearLoss;
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
        float mult = - loss / (a * matrix.GetLength(0) * matrix.GetLength(1));

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

    private static Vector[] CalculateGradient(float[,] matrix, Vector[] gradientVectors, Vector[] dotVectors, float loss)
    {
        int length = gradientVectors.Length;
        Vector[] gradients = new Vector[length];
        for(int i = 0; i < length; i++)
        {
            gradients[i] = new Vector(gradientVectors[i].Length);
        }
        for (int i = 1; i < length; i++)
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

    private (Vector[], Vector[]) CalculateGradient(float[,] matrix, float loss)
    {
        return (CalculateGradient(matrix, _imageVectors, _descriptionVectors, loss),
            CalculateGradient(TransposeArray(matrix), _descriptionVectors, _imageVectors, loss));
        /*(Vector[], Vector[]) gradients = (new Vector[_batchSize], new Vector[_batchSize]);
        for (int i = 0; i < _batchSize; i++)
        {
            float totalI = ASYMPTOTEERRORFACTOR;
            float totalD = ASYMPTOTEERRORFACTOR;
            Vector dI_dV = -_descriptionVectors[i];
            Vector dD_dV = -_imageVectors[i];
            for (int j = 0; j < _batchSize; j++)
            {
                totalI += matrix[i, j];
                totalD += matrix[j, i];

                if (i != j)
                {
                    if (matrix[i, j] > 0)
                    {
                        dI_dV.Subtract(_descriptionVectors[j]);
                    }
                    if (matrix[j, i] > 0)
                    {
                        dD_dV.Subtract(_imageVectors[j]);
                    }
                }
            }

            dI_dV.Mult(1 / totalI);
            dD_dV.Mult(1 / totalD);

            float inv = 1 / (matrix[i, i] + ASYMPTOTEERRORFACTOR);

            dI_dV.Add(_descriptionVectors[i]* inv);
            dD_dV.Add(_imageVectors[i] * inv);

            float batchInv = loss / _batchSize;

            dI_dV.Mult(-batchInv);
            dD_dV.Mult(-batchInv);

            gradients.Item1[i] = dI_dV;
            gradients.Item2[i] = dD_dV;
        }
        return gradients;
    }*/
    }
}