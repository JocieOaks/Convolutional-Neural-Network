// See https://aka.ms/new-console-template for more information
#nullable disable

using Newtonsoft.Json;
using System.Runtime.Serialization;

[Serializable]
public class ConvolutionalLayer : Layer
{
    [JsonProperty]
    protected Color[][,] _kernals;

    float _invK2;

    public ConvolutionalLayer(int kernalsNum, int kernalSize, int stride) : base(kernalsNum, kernalSize, stride)
    {
        _kernals = new Color[kernalsNum][,];

        for (int i = 0; i < kernalsNum; i++)
        {
            _kernals[i] = new Color[_kernalSize, _kernalSize];
            for (int j = 0; j < kernalSize; j++)
            {
                for (int k = 0; k < kernalSize; k++)
                {
                    _kernals[i][j, k] = Color.Random(0.5f);
                }
            }
        }
        _invK2 = 1f / (kernalSize * kernalSize);
    }

    public ConvolutionalLayer() : base(0, 0, 0) { }

    [OnDeserialized]
    public void OnDeserialized(StreamingContext context)
    {
        _invK2 = 1f / (_kernalSize * _kernalSize);
    }

    public override FeatureMap[][] Backwards(FeatureMap[][] dL_dP, FeatureMap[][] input, float alpha)
    {
        FeatureMap[][] dL_dPNext = new FeatureMap[_dimensions][];
        for (int i = 0; i < _dimensions; i++)
        {
            dL_dPNext[i] = Backwards(dL_dP[i], input[i], _kernals[i], alpha);
        }

        return dL_dPNext;
    }

    public override FeatureMap[][] Forward(FeatureMap[][] input)
    {
        FeatureMap[][] convoluted = new FeatureMap[_dimensions][];
        for (int i = 0; i < _dimensions; i++)
        {
            convoluted[i] = Forward(input[i], _kernals[i]);
        }
        return convoluted;
    }

    protected FeatureMap[] Backwards(FeatureMap[] dL_dP, FeatureMap[] input, Color[,] kernal, float alpha)
    {
        int batchLength = input.Length;
        FeatureMap[] dL_dPNext = new FeatureMap[batchLength];

        for (int i = 0; i < batchLength; i++)
        {
            dL_dPNext[i] = Backwards(dL_dP[i], input[i], kernal, alpha);
        }

        return dL_dPNext;
    }

    protected FeatureMap Backwards(FeatureMap dL_dP, FeatureMap input, Color[,] kernal, float alpha)
    {
        int widthSubdivisions = dL_dP.Width;
        int lengthSubdivisions = dL_dP.Length;
        FeatureMap dL_dPNext = new FeatureMap(input.Width, input.Length);
        Color[,] dL_dK = new Color[_kernalSize, _kernalSize];
        for (int strideX = 0; strideX < widthSubdivisions; strideX++)
        {
            for (int strideY = 0; strideY < lengthSubdivisions; strideY++)
            {
                dL_dPNext[strideX, strideY] = new();
                for (int kernalX = 0; kernalX < _kernalSize; kernalX++)
                {
                    for (int kernalY = 0; kernalY < _kernalSize; kernalY++)
                    {
                        int x = strideX * _stride + kernalX;
                        int y = strideY * _stride + kernalY;
                        Color dK = dL_dP[strideX, strideY]* input[x, y] * _invK2;
                        Color dP = dL_dP[strideX, strideX] * kernal[kernalX, kernalY] * _invK2;
                        dL_dK[kernalX, kernalY] += dK;
                        dL_dK[kernalX, kernalY].Clamp(1);
                        dL_dPNext[x, y] += dP;
                    }
                }
            }
        }

        for(int i = 0; i < dL_dPNext.Width; i++)
        {
            for(int j = 0; j < dL_dPNext.Length; j++)
            {
                dL_dPNext[i, j] = dL_dPNext[i, j].Clamp(0.5f);
            }
        }

        for (int i = 0; i < _kernalSize; i++)
        {
            for (int j = 0; j < _kernalSize; j++)
            {
                kernal[i, j] -= alpha * dL_dK[i, j];
            }
        }

        return dL_dPNext;
    }

    protected FeatureMap[] Forward(FeatureMap[] input, Color[,] kernal)
    {
        Pad(input);
        FeatureMap[] convoluted = new FeatureMap[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            convoluted[i] = Forward(input[i], kernal);
        }
        
        return convoluted;
    }

    protected FeatureMap Forward(FeatureMap input, Color[,] kernal)
    {
        int widthSubdivisions = (input.Width - _kernalSize) / _stride;
        int lengthSubdivisions = (input.Length - _kernalSize) / _stride;
        FeatureMap convoluted = new FeatureMap(widthSubdivisions, lengthSubdivisions);
        for (int shiftX = 0; shiftX < widthSubdivisions; shiftX++)
        {
            for (int shiftY = 0; shiftY < lengthSubdivisions; shiftY++)
            {
                for (int kernalX = 0; kernalX < _kernalSize; kernalX++)
                {
                    for (int kernalY = 0; kernalY < _kernalSize; kernalY++)
                    {
                        convoluted[shiftX, shiftY] += input[shiftX * _stride + kernalX, shiftY * _stride + kernalY] * kernal[kernalX, kernalY];
                    }
                }
                convoluted[shiftX, shiftY] *= _invK2;
            }
        }

        return convoluted;
    }
}
