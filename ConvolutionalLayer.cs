// See https://aka.ms/new-console-template for more information
#nullable disable

public class ConvolutionalLayer<T> : Layer<T> where T : IDot<T>, new()
{
    protected int _kernalNum;

    protected T[,,] _kernals;
    T[,] _currentFeatures;

    public ConvolutionalLayer(int kernalsNum, int kernalSize, int stride) : base(kernalSize, stride)
    {
        _kernalNum = kernalsNum;
        _kernals = new T[kernalsNum, kernalSize, kernalSize];

        for (int i = 0; i < kernalsNum; i++)
        {
            for (int j = 0; j < kernalSize; j++)
            {
                for (int k = 0; k < kernalSize; k++)
                {
                    _kernals[i, j, k] = new T().Random();
                }
            }
        }
    }

    protected T[,] Forward(T[,] input, int kernalIndex)
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
                        convoluted[shiftX, shiftY] = convoluted[shiftX, shiftY].Add(input[shiftX * _stride + kernalX, shiftY * _stride + kernalY].Multiply(_kernals[kernalIndex, kernalX, kernalY]));
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
            convoluted[i] = Forward(input[i], i);
        }
        return convoluted;
    }

    public override T[][,] Backwards(T[][,] error, float alpha)
    {
        T[][,] corrections = new T[_kernalNum][,];
        int widthSubdivisions = error.GetLength(1);
        int lengthSubdivisions = error.GetLength(2);
        for (int kernalIndex = 0; kernalIndex < _kernalNum; kernalIndex++)
        {
            corrections[kernalIndex] = new T[_currentFeatures.GetLength(0), _currentFeatures.GetLength(1)];
            for(int strideX = 0; strideX < widthSubdivisions; strideX++)
            {
                for(int strideY = 0; strideY < lengthSubdivisions; strideY++)
                {
                    corrections[kernalIndex][strideX, strideY] = new();
                    for(int kernalX = 0;  kernalX < _kernalSize;  kernalX++)
                    {
                        for(int kernalY = 0; kernalY < _kernalSize; kernalY++)
                        {
                            T pointError = error[kernalIndex][strideX, strideY].Multiply(_currentFeatures[strideX * _stride + kernalX, strideY * _stride + kernalY]);
                            corrections[kernalIndex][strideX * _stride + kernalX, strideY * _stride + kernalY] = pointError;
                            _kernals[kernalIndex, kernalX, kernalY] = _kernals[kernalIndex, kernalX, kernalY].Subtract(pointError.Multiply(alpha));
                        }
                    }
                }
            }
        }

        return corrections;
    }
}
