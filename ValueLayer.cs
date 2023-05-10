// See https://aka.ms/new-console-template for more information
public class ValueLayer
{
    DotFloat[,] Forward(Color[,] input)
    {
        DotFloat[,] values = new DotFloat[input.GetLength(0), input.GetLength(1)];
        for(int i = 0; i < input.GetLength(0); i++)
        {
            for(int j = 0; j < input.GetLength(1); j++)
            {
                values[i, j] = new DotFloat(input[i, j].Magnitude);
            }
        }

        return values;
    }

    public DotFloat[][,] Forward(Color[][,] input)
    {
        DotFloat[][,] values = new DotFloat[input.Length][,];
        for(int i = 0; i < input.Length; i++)
        {
            values[i] = Forward(input[i]);
        }
        return values;
    }

    public Color[,] Backwards(DotFloat[,] error)
    {
        throw new NotImplementedException();

        Color[,] corrections = new Color[error.GetLength(0), error.GetLength(1)];
        for(int i = 0; i <error.GetLength(0); i++)
        {
            for(int j = 0; j <error.GetLength(1); j++)
            {
                //corrections[i, j] = _currentFeatures[i, j].Dot(error[i, j].Value / _values[i, j].Value);
            }
        }

        return corrections;
    }
}
