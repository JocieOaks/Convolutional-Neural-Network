// See https://aka.ms/new-console-template for more information
using System.Runtime.Serialization;

[Serializable]
public class AveragePoolLayer<T> : Layer<T> where T : IDot<T>, new()
{
    float _inverseKernal2;

    public AveragePoolLayer(int kernalSize) : base(kernalSize,kernalSize)
    {
        _inverseKernal2 = 1f / (kernalSize * kernalSize);
    }

    [OnDeserialized]
    public void OnDeserialized(StreamingContext context)
    {
        _inverseKernal2 = 1f / (_kernalSize * _kernalSize);
    }

    public override T[][,] Forward(T[][,] input)
    {
        Pad(input);
        T[][,] pooled = new T[input.Length][,];
        for(int i = 0; i < input.Length; i++)
        {
            pooled[i] = Forward(input[i]);
        }

        return pooled;
    }

    T[,] Forward(T[,] input)
    {
        int widthSubdivisions = input.GetLength(0) / _kernalSize;
        int lengthSubdivisions = input.GetLength(1) / _kernalSize;
        T[,] pooled = new T[widthSubdivisions, lengthSubdivisions];
        for(int i =  0; i < widthSubdivisions; i++)
        {
            for(int j = 0; j < lengthSubdivisions; j++)
            {
                for(int  k = 0; k < _kernalSize; k++)
                {
                    for(int l = 0; l < _kernalSize; l++)
                    {
                        pooled[i, j] = pooled[i, j].Add(input[i * _kernalSize + k, j * _kernalSize + l]);
                    }
                }
                pooled[i, j] = pooled[i, j].Multiply(0.25f);
            }
        }
        return pooled;
    }

    public override T[][,] Backwards(T[][,] dL_dP, T[][,] input, float _)
    {
        T[][,] dL_dPNext = new T[dL_dP.Length][,];
        for(int i = 0; i < dL_dP.Length; i++)
        {
            dL_dPNext[i] = Backwards(dL_dP[i]);
        }
        return dL_dPNext;
    }

    T[,] Backwards(T[,] dL_dP)
    {
        int width = dL_dP.GetLength(0) * _kernalSize;
        int length = dL_dP.GetLength(1) * _kernalSize;
        T[,] dL_dPNext = new T[width, length];

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < length; y++)
            {
                dL_dPNext[x, y] = dL_dP[x / _kernalSize, y / _kernalSize].Multiply(_inverseKernal2);
            }
        }

        return dL_dPNext;
    }
}
