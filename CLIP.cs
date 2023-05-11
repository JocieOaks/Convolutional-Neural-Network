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
    readonly private InitialConvolutionLayer<Color> _initialConvolutionLayer;
    [JsonProperty]
    readonly private List<Layer<Color>> _layers = new();

    private Color[][][,] _initialFeatureMaps;
    private Color[,][][,] _featureMaps;
    private Vector[] _imageVectors;
    private Vector[] _descriptionVectors;

    private int Depth => _layers.Count;
    [JsonProperty]
    readonly private int _batchSize;

    public CLIP(int depth, int kernals, int vectorDimensions, int batchSize, int descriptionLength)
    {
        _initialConvolutionLayer = new InitialConvolutionLayer<Color>(kernals, 3, 2);
        AveragePoolLayer<Color> sumPoolLayer = new(2);
        _vectorizationLayer = new VectorizationLayer(vectorDimensions, kernals);
        _transformer = new Transformer(descriptionLength, vectorDimensions);
        _batchSize = batchSize;

        _imageVectors = new Vector[batchSize];
        _descriptionVectors = new Vector[batchSize];

        for (int i = 0; i < depth; i++)
        {
            _layers.Add(new ConvolutionalLayer<Color>(kernals, 3, 1));
            _layers.Add(new NormalizationLayer<Color>(kernals));
            if (i < 9 && i % 3 == 0)
                _layers.Add(sumPoolLayer);
        }
        _initialFeatureMaps = new Color[batchSize][][,];
        _featureMaps = new Color[batchSize, Depth][][,];
    }

    public CLIP() { }

    [OnDeserialized]
    public void OnDeserialized(StreamingContext context)
    {
        _imageVectors = new Vector[_batchSize];
        _descriptionVectors = new Vector[_batchSize];
        _initialFeatureMaps = new Color[_batchSize][][,];
        _featureMaps = new Color[_batchSize, Depth][][,];
    }

    public void Forward((Color[,] image, int[] vector)[] _input)
    {
        Color[][,] current;
        for(int i = 0; i < _batchSize; i++)
        {
            current = _initialConvolutionLayer.Forward(_input[i].image);
            _initialFeatureMaps[i] = current;
            for(int j = 0; j < Depth; j++)
            {
                current = _layers[j].Forward(current);
                _featureMaps[i, j] = current;
            }

            _imageVectors[i] = _vectorizationLayer.Forward(_featureMaps[i, Depth - 1]).Normalized();
            _descriptionVectors[i] = _transformer.Forward(_input[i].vector).Normalized();
        }
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

    public void Backwards((Vector dL_dI, Vector dL_dD)[] gradients, (Color[,] image, int[] vector)[] input, float alpha)
    {
        for(int i = 0; i < _batchSize; i++)
        {
            _transformer.Backwards(gradients[i].dL_dD, input[i].vector, alpha);
            Color[][,] current = _vectorizationLayer.Backwards(gradients[i].dL_dI, _imageVectors[i], _featureMaps[i, Depth - 1], alpha);
            for (int j = Depth - 1; j > 0; j--)
            {
                current = _layers[j].Backwards(current, _featureMaps[i, j - 1], alpha);
            }
            current = _layers[0].Backwards(current, _initialFeatureMaps[i], alpha);
            _initialConvolutionLayer.Backwards(current, input[i].image, alpha);
        }
    }

    public float Train((Color[,] image, int[] vector)[] input, float alpha)
    {
        Forward(input);
        float[,] matrix = Score();
        float loss = Loss(matrix);
        (Vector, Vector)[] gradients = CalculateGradient(matrix, loss);
        Backwards(gradients, input, alpha);
        return loss;
    }

    (Vector, Vector)[] CalculateGradient(float[,] matrix, float loss)
    {
        (Vector, Vector)[] gradients = new (Vector, Vector)[_batchSize];
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
                        dI_dV -= _descriptionVectors[j];
                    }
                    if (matrix[j, i] > 0)
                    {
                        dD_dV -= _imageVectors[j];
                    }
                }
            }

            dI_dV *= 1 / totalI;
            dD_dV *= 1 / totalD;

            float inv = 1 / matrix[i, i];

            dI_dV += _descriptionVectors[i]* inv;
            dD_dV += _imageVectors[i] * inv;

            dI_dV *= -loss / _batchSize;
            dD_dV *= -loss / _batchSize;

            gradients[i] = (dI_dV, dD_dV);
        }
        return gradients;
    }
}