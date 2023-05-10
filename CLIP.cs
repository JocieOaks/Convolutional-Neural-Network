// See https://aka.ms/new-console-template for more information
using System.Text.Json.Serialization;

public class CLIP
{
    readonly private VectorizationLayer _vectorizationLayer;
    readonly private ValueLayer _valueLayer = new();
    readonly private Transformer _transformer;
    readonly private FirstConvolutionalLayer<Color> _firstConvolutionalLayer;
    readonly private List<Layer<Color>> _layers = new();

    readonly private Color[][][,] _firstFeatureMaps;
    readonly private Color[,][][,] _featureMaps;
    readonly private (Color[,] image, int[] vector)[] _input;
    readonly private DotFloat[][][,] _valueFeatureMaps;
    readonly private Vector[] _imageVectors;
    readonly private Vector[] _transformedVectors;

    readonly private int _depth;
    readonly private int _vectorDimensions;
    readonly private int _batchSize;

    public CLIP(int depth, int kernals, int vectorDimensions, (Color[,], int[])[] input)
    {
        _firstConvolutionalLayer = new FirstConvolutionalLayer<Color>(kernals, 3, 1);
        AveragePoolLayer<Color> sumPoolLayer = new AveragePoolLayer<Color>(2);
        _vectorizationLayer = new VectorizationLayer(vectorDimensions);
        _transformer = new Transformer(input[0].Item2.Length, vectorDimensions);
        _input = input;
        _batchSize = input.Length;
        _vectorDimensions = vectorDimensions;

        

        _valueFeatureMaps = new DotFloat[_batchSize][][,];
        _imageVectors = new Vector[_batchSize];
        _transformedVectors = new Vector[_batchSize];

        for(int i = 0; i < depth; i++)
        {
            _layers.Add(new ConvolutionalLayer<Color>(kernals, 3, 1));
            if (i < 8 && i % 2 == 0)
                _layers.Add(sumPoolLayer);
        }
        _depth = _layers.Count;
        _firstFeatureMaps = new Color[_batchSize][][,];
        _featureMaps = new Color[_batchSize, _depth][][,];
    }

    public float[,] Forward()
    {
        Color[][,] current;
        for(int i = 0; i < _batchSize; i++)
        {
            current = _firstConvolutionalLayer.Forward(_input[i].image);
            _firstFeatureMaps[i] = current;
            Console.WriteLine("Initializing Layer Complete");
            for(int j = 0; j < _depth; j++)
            {
                current = _layers[j].Forward(current);
                _featureMaps[i, j] = current;
                Console.WriteLine(j + " Layer Complete");
            }

            _valueFeatureMaps[i] = _valueLayer.Forward(_featureMaps[i, _depth - 1]);
            Console.WriteLine("Value Layer Complete");
            _imageVectors[i] = _vectorizationLayer.Forward(_valueFeatureMaps[i]);
            Console.WriteLine("Vectorization Complete");
            _transformedVectors[i] = _transformer.Forward(_input[i].vector);
            Console.WriteLine("Vector Transformation Complete");
        }

        float[,] cosScores = new float[_batchSize, _batchSize];

        for(int i = 0; i < _batchSize; i++)
        {
            cosScores[i,i] = Vector.Cos(_imageVectors[i], _transformedVectors[i]);
            for(int j = i + 1; j < _batchSize; j++)
            {
                float score = Vector.Cos(_imageVectors[i], _imageVectors[j]);
                cosScores[i, j] = score;
                cosScores[j, i] = score;
            }
        }

        return cosScores;
    }

    public void Backward(float[,] output)
    {
        for(int i = 0; i < _batchSize; i++)
        {
            Vector transformError = new Vector(_vectorDimensions);
            Vector imageError = new Vector(_vectorDimensions);
            for(int j = 0; j < _batchSize; j++)
            {
                if(i == j)
                {
                    
                    Vector error = _imageVectors[i] - _transformedVectors[i];
                    transformError += error;
                    imageError -= error;
                }
                else
                {
                    transformError -= _transformedVectors[i] + _imageVectors[j];
                    imageError -= _imageVectors[i] + _transformedVectors[j];
                }
            }

            _transformer.Backward(transformError, _transformedVectors[i], 0.05f);
        }
    }
}