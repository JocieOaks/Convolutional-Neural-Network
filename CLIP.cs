// See https://aka.ms/new-console-template for more information

using Newtonsoft.Json;
using System.Runtime.Serialization;

[Serializable]
public class CLIP
{
    public static Random Random { get; } = new Random();

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

    public CLIP(int depth, int dimensions, int vectorDimensions, int batchSize, int descriptionLength)
    {
        _initialConvolutionLayer = new InitialConvolutionLayer(dimensions, 3, 2);
        AveragePoolLayer sumPoolLayer = new(dimensions, 2);
        _vectorizationLayer = new VectorizationLayer(vectorDimensions, dimensions);
        _transformer = new Transformer(descriptionLength, vectorDimensions);
        _batchSize = batchSize;

        _imageVectors = new Vector[batchSize];
        _descriptionVectors = new Vector[batchSize];

        for (int i = 0; i < depth; i++)
        {
            _layers.Add(new ConvolutionalLayer(dimensions, 3, 1));
            _layers.Add(new NormalizationLayer(dimensions));
            if (i < 6 && i % 2 == 0)
                _layers.Add(sumPoolLayer);
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

    public void Forward((FeatureMap image, int[] vector)[] input)
    {
        FeatureMap[][] current;
        FeatureMap[] images = new FeatureMap[_batchSize];
        for(int i = 0; i < _batchSize; i++)
        {
            images[i] = input[i].image; 
            _descriptionVectors[i] = _transformer.Forward(input[i].vector).Normalized();
        }

        current = _initialConvolutionLayer.Forward(images);
        _initialFeatureMaps = current;

        for(int j = 0; j < Depth; j++)
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
        for(int i = 0; i < transposed.Length; i++)
        {
            transposed[i] = new T[array.Length];
            for (int j = 0; j < transposed[i].Length; j++)
            {
                transposed[i][j] = array[j][i];
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
                cosScores[i, j] = MathF.Max(Vector.Dot(_imageVectors[i], _descriptionVectors[j]), i == j ? 0.01f : 0);
            }
        }

        return cosScores;
    }

    public static float Loss(float[,] matrix)
    {
        float loss = 0.0f;
        int length = matrix.GetLength(0);
        for(int i = 0; i < length; i++)
        {
            float totalI = 0;
            float totalD = 0;
            for(int j = 0; j < length; j++)
            {
                totalI += matrix[i, j];
                totalD += matrix[j, i];
            }

            for(int j = 0; j < length; j++)
            {
                if (i == j)
                    loss += -1f / length * MathF.Log(matrix[i, j] * matrix[j, i] / totalD / totalI);
                else
                    loss += -1f / length * (MathF.Log((totalD - matrix[j, i]) / totalD) + MathF.Log((totalI - matrix[i, j]) / totalI));
            }
        }
        return loss;
    }

    public void Backwards((Vector[] dL_dI, Vector[] dL_dD) gradients, (FeatureMap image, int[] vector)[] input, float alpha)
    {
        FeatureMap[] images = new FeatureMap[_batchSize];
        for (int i = 0; i < _batchSize; i++)
        {
            images[i] = input[i].image;
            _transformer.Backwards(gradients.dL_dD[i], input[i].vector, alpha);
        }

        FeatureMap[][] transposed = new FeatureMap[_batchSize][];
        for (int i = 0; i < _batchSize; i++)
        {
            transposed[i] = _vectorizationLayer.Backwards(gradients.dL_dI[i], _imageVectors[i], _transposedFinalFeatureMap[i], alpha);
        }

        FeatureMap[][] current = TransposeArray(transposed);

        for (int j = Depth - 1; j > 0; j--)
        {
            current = _layers[j].Backwards(current, _featureMaps[j - 1], alpha);
        }
        current = _layers[0].Backwards(current, _initialFeatureMaps, alpha);
        _initialConvolutionLayer.Backwards(current, images, alpha);
    }

    public float Train((FeatureMap image, int[] vector)[] input, float alpha)
    {
        Forward(input);
        float[,] matrix = Score();
        float loss = Loss(matrix);
        (Vector[], Vector[]) gradients = CalculateGradient(matrix, loss);
        Backwards(gradients, input, alpha);
        return loss;
    }

    (Vector[], Vector[]) CalculateGradient(float[,] matrix, float loss)
    {
        (Vector[], Vector[]) gradients = (new Vector[_batchSize], new Vector[_batchSize]);
        for (int i = 0; i < _batchSize; i++)
        {
            float totalI = 0;
            float totalD = 0;
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

            float inv = 1 / matrix[i, i];

            dI_dV.Add(_descriptionVectors[i]* inv);
            dD_dV.Add(_imageVectors[i] * inv);

            float batchInv = loss / _batchSize;

            dI_dV.Mult(-batchInv);
            dD_dV.Mult(-batchInv);

            gradients.Item1[i] = dI_dV;
            gradients.Item2[i] = dD_dV;
        }
        return gradients;
    }
}