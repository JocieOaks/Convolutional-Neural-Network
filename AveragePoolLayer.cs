// See https://aka.ms/new-console-template for more information
using System.Runtime.Serialization;

[Serializable]
public class AveragePoolLayer : Layer
{
    float _invK2;

    public AveragePoolLayer(int dimensions,int kernalSize) : base(dimensions, kernalSize, kernalSize)
    {
        _invK2 = 1f / (kernalSize * kernalSize);
    }

    public AveragePoolLayer() : base(0, 0, 0) { }

    public override FeatureMap[][] Backwards(FeatureMap[][] dL_dP, FeatureMap[][] input, float alpha)
    {
        FeatureMap[][] dL_dPNext = new FeatureMap[_dimensions][];
        for(int i = 0; i < _dimensions; i++)
        {
            dL_dPNext[i] = Backwards(dL_dP[i]);
        }

        return dL_dPNext;
    }

    public override FeatureMap[][] Forward(FeatureMap[][] input)
    {
        FeatureMap[][] pooled = new FeatureMap[_dimensions][];
        for(int i = 0; i < _dimensions; i++)
        {
            pooled[i] = Forward(input[i]);
        }
        return pooled;
    }

    [OnDeserialized]
    public void OnDeserialized(StreamingContext context)
    {
        _invK2 = 1f / (_kernalSize * _kernalSize);
    }

    FeatureMap[] Backwards(FeatureMap[] dL_dP)
    {
        FeatureMap[] dL_dPNext = new FeatureMap[dL_dP.Length];
        for (int i = 0; i < dL_dP.Length; i++)
        {
            dL_dPNext[i] = Backwards(dL_dP[i]);
        }
        return dL_dPNext;
    }

    FeatureMap Backwards(FeatureMap dL_dP)
    {
        int width = dL_dP.Width * _kernalSize;
        int length = dL_dP.Length * _kernalSize;
        FeatureMap dL_dPNext = new FeatureMap(width, length);

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < length; y++)
            {
                dL_dPNext[x, y] = dL_dP[x / _kernalSize, y / _kernalSize] * _invK2;
            }
        }

        return dL_dPNext;
    }

    private FeatureMap[] Forward(FeatureMap[] input)
    {
        Pad(input);
        FeatureMap[] pooled = new FeatureMap[input.Length];
        for(int i = 0; i < input.Length; i++)
        {
            pooled[i] = Forward(input[i]);
        }

        return pooled;
    }

    FeatureMap Forward(FeatureMap input)
    {
        int widthSubdivisions = input.Width / _kernalSize;
        int lengthSubdivisions = input.Length / _kernalSize;
        FeatureMap pooled = new FeatureMap(widthSubdivisions, lengthSubdivisions);
        for(int i =  0; i < widthSubdivisions; i++)
        {
            for(int j = 0; j < lengthSubdivisions; j++)
            {
                for(int  k = 0; k < _kernalSize; k++)
                {
                    for(int l = 0; l < _kernalSize; l++)
                    {
                        pooled[i, j] += input[i * _kernalSize + k, j * _kernalSize + l];
                    }
                }
                pooled[i, j] *= 0.25f;
            }
        }
        return pooled;
    }
}
