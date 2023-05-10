using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

public abstract class Layer<T> where T : IDot<T>, new()
{
    protected int _kernalSize;
    protected int _stride;

    public Layer(int kernalSize, int stride)
    {
        _kernalSize = kernalSize;
        _stride = stride;
    }

    protected void Pad(T[][,] input)
    {
        int paddingX = (input[0].GetLength(0) - _kernalSize) % _stride;
        int paddingY = (input[0].GetLength(1) - _kernalSize) % _stride;
        if (paddingX == 0 && paddingY == 0)
            return;

        int halfX = paddingX / 2;
        int halfY = paddingY / 2;
        for (int k = 0; k < input.Length; k++)
        {
            T[,] padded = new T[input[k].GetLength(0) + paddingX, input[k].GetLength(1) + paddingY];
            for (int i = 0; i < input[k].GetLength(0); i++)
            {
                for (int j = 0; j < input[k].GetLength(1); j++)
                {
                    padded[i + halfX, j + halfY] = input[k][i, j];
                }
            }
            input[k] = padded;
        }
        return;
    }

    public abstract T[][,] Forward(T[][,] input);

    public abstract T[][,] Backwards(T[][,] error, float alpha);
}

