// See https://aka.ms/new-console-template for more information
#nullable disable

using Newtonsoft.Json;

[Serializable]
public class ConvolutionalLayer<T> : Layer<T> where T : IDot<T>, new()
{
    [JsonProperty]
    protected int _kernalNum;

    [JsonProperty]
    protected T[][,] _kernals;

    public ConvolutionalLayer(int kernalsNum, int kernalSize, int stride) : base(kernalSize, stride)
    {
        _kernalNum = kernalsNum;
        _kernals = new T[kernalsNum][,];

        for (int i = 0; i < kernalsNum; i++)
        {
            _kernals[i] = new T[_kernalSize, _kernalSize];
            for (int j = 0; j < kernalSize; j++)
            {
                for (int k = 0; k < kernalSize; k++)
                {
                    _kernals[i][j, k] = new T().Random();
                }
            }
        }
    }

    [JsonConstructor]
    public ConvolutionalLayer(int kernalSize, int stride) : base(kernalSize, stride) { }

    protected T[,] Forward(T[,] input, T[,] kernal)
    {
        int widthSubdivisions = (input.GetLength(0) - _kernalSize) / _stride;
        int lengthSubdivisions = (input.GetLength(1) - _kernalSize) / _stride;
        T[,] convoluted = new T[widthSubdivisions, lengthSubdivisions];
        for (int shiftX = 0; shiftX < widthSubdivisions; shiftX++)
        {
            for (int shiftY = 0; shiftY < lengthSubdivisions; shiftY++)
            {
                for (int kernalX = 0; kernalX < _kernalSize; kernalX++)
                {
                    for (int kernalY = 0; kernalY < _kernalSize; kernalY++)
                    {
                        convoluted[shiftX, shiftY] = convoluted[shiftX, shiftY].Add(input[shiftX * _stride + kernalX, shiftY * _stride + kernalY].Multiply(kernal[kernalX, kernalY]));
                    }
                }
                convoluted[shiftX, shiftY] = convoluted[shiftX, shiftY].Multiply(1f / (_kernalSize * _kernalSize));
            }
        }
        
        return convoluted;
    }

    public override T[][,] Forward(T[][,] input)
    {
        Pad(input);
        T[][,] convoluted = new T[_kernalNum][,];
        for (int i = 0; i < _kernalNum; i++)
        {
            convoluted[i] = Forward(input[i], _kernals[i]);
        }
        return convoluted;
    }

    public override T[][,] Backwards(T[][,] dL_dP, T[][,] input, float alpha)
    {
        T[][,] dL_dPNext = new T[_kernalNum][,];
        
        for (int i = 0; i < _kernalNum; i++)
        {
            dL_dPNext[i] = Backwards(dL_dP[i], input[i], _kernals[i], alpha);
        }

        return dL_dPNext;
    }

    protected T[,] Backwards(T[,] dL_dP, T[,] input, T[,] kernal, float alpha)
    {
        int widthSubdivisions = dL_dP.GetLength(0);
        int lengthSubdivisions = dL_dP.GetLength(1);
        T[,] dL_dPNext = new T[input.GetLength(0), input.GetLength(1)];
        T[,] dL_dK = new T[_kernalSize, _kernalSize];
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
                        T dK = dL_dP[strideX, strideY].Multiply(input[x,y]);
                        T dP = dL_dP[strideX, strideX].Multiply(kernal[kernalX, kernalY]);
                        dL_dK[kernalX, kernalY] = dK;
                        dL_dPNext[x, y] = dL_dPNext[x, y].Add(dP);
                    }
                }
            }
        }

        for(int i = 0; i < _kernalSize; i++)
        {
            for(int j = 0; j < _kernalSize; j++)
            {
                kernal[i, j] = kernal[i, j].Subtract(dL_dK[i, j].Multiply(alpha));
            }
        }

        return dL_dPNext;
    }
}
