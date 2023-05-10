// See https://aka.ms/new-console-template for more information
public class AveragePoolLayer<T> : Layer<T> where T : IDot<T>, new()
{

    public AveragePoolLayer(int kernalSize) : base(kernalSize,kernalSize)
    {
    }

    public override T[][,] Forward(T[][,] input)
    {
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
                        pooled[i, j] = pooled[i, j].Add(input[i * _kernalSize + k, j * _kernalSize + l].ReLU());
                    }
                }
                pooled[i, j] = pooled[i, j].Multiply(0.25f);
            }
        }
        return pooled;
    }

    public override T[][,] Backwards(T[][,] error, float alpha)
    {

        throw new NotImplementedException();

        int widthSubdivisions = error.GetLength(1);
        int lengthSubdivisions = error.GetLength(2);
        T[,] corrections = new T[widthSubdivisions, lengthSubdivisions];

        for (int shiftX = 0; shiftX < widthSubdivisions; shiftX++)
        {
            for (int shiftY = 0; shiftY < lengthSubdivisions; shiftY++)
            {
                for (int kernalX = 0; kernalX < _kernalSize; kernalX++)
                {
                    for (int kernalY = 0; kernalY < _kernalSize; kernalY++)
                    {
                        //corrections[shiftX * _kernalSize + kernalX, shiftY * _kernalSize + kernalY] = error[shiftX, shiftY].Dot(_pooled[shiftX, shiftY].Divide(_currentFeatures[shiftX * _kernalSize + kernalX, shiftY * _kernalSize + kernalY].ReLU()));
                    }
                }
            }
        }

        //return corrections;
    }
}
