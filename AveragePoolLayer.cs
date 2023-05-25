// See https://aka.ms/new-console-template for more information
using System.Runtime.Serialization;

[Serializable]
public class AveragePoolLayer : Layer
{
    private readonly static float FOURTH = 0.25f;
    float _invK2;
    private readonly FeatureMap[][] _pooled;
    private readonly FeatureMap[][] _dL_dPNext;
    public AveragePoolLayer(int kernalSize, ref FeatureMap[][] input) : base(input.Length, kernalSize, kernalSize)
    {
        _invK2 = 1f / (kernalSize * kernalSize);
        _pooled = new FeatureMap[_dimensions][];
        _dL_dPNext = new FeatureMap[_dimensions][];
        for(int i = 0; i < _dimensions; i++)
        {
            Pad(input[i]);
            _pooled[i] = new FeatureMap[input[i].Length];
            _dL_dPNext[i] = new FeatureMap[input[i].Length];
            for(int j = 0; j < input[i].Length; j++)
            {
                _dL_dPNext[i][j] = new FeatureMap(input[i][j].Width, input[i][j].Length);
                int width = input[i][j].Width / kernalSize;
                int length = input[i][j].Length / kernalSize;
                _pooled[i][j] = new FeatureMap(width, length);
            }
        }
        input = _pooled;
    }

    public AveragePoolLayer() : base(0, 0, 0) { }

    public override FeatureMap[][] Backwards(FeatureMap[][] input, FeatureMap[][] dL_dP, float learningRate)
    {
        for(int i = 0; i < _dimensions; i++)
        {
            Backwards(dL_dP[i], _dL_dPNext[i]);
        }

        return _dL_dPNext;
    }

    public override FeatureMap[][] Forward(FeatureMap[][] input)
    {
        for(int i = 0; i < _dimensions; i++)
        {
            Forward(input[i], _pooled[i]);
        }
        return _pooled;
    }

    [OnDeserialized]
    public void OnDeserialized(StreamingContext context)
    {
        _invK2 = 1f / (_kernalSize * _kernalSize);
    }

    void Backwards(FeatureMap[] dL_dP, FeatureMap[] dL_dPNext)
    {
        for (int i = 0; i < dL_dP.Length; i++)
        {
            Backwards(dL_dP[i], dL_dPNext[i]);
        }
    }

    private void Backwards(FeatureMap dL_dP, FeatureMap dL_dPNext)
    {

        for (int y = 0; y < dL_dPNext.Length; y++)
        {
            for (int x = 0; x < dL_dPNext.Width; x++)
            {
                dL_dPNext[x, y] = dL_dP[x / _kernalSize, y / _kernalSize] * _invK2;
            }
        }
    }

    private void Forward(FeatureMap[] input, FeatureMap[] pooled)
    {
        for(int i = 0; i < input.Length; i++)
        {
            Forward(input[i], pooled[i]);
        }
    }

    private void Forward(FeatureMap input, FeatureMap pooled)
    {
        for(int i =  0; i < pooled.Width; i++)
        {
            for(int j = 0; j < pooled.Length; j++)
            {
                for(int  k = 0; k < _kernalSize; k++)
                {
                    for(int l = 0; l < _kernalSize; l++)
                    {
                        pooled[i, j] += input[i * _kernalSize + k, j * _kernalSize + l];
                    }
                }
                pooled[i, j] *= FOURTH;
            }
        }
    }
}
