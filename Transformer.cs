// See https://aka.ms/new-console-template for more information
public class Transformer
{
    float[,] _matrix;

    public Transformer(int inputs, int outputs)
    {
        _matrix = new float[outputs, inputs];
        for(int i = 0; i < outputs; i++)
        {
            for(int j = 0; j < inputs; j++)
            {
                _matrix[i, j] = (float)new Random().NextDouble();
            }
        }
    }

    public Vector Forward(int[] input)
    {
        float[] output = new float[_matrix.GetLength(0)];
        for(int i = 0; i < _matrix.GetLength(0); i++)
        {
            for(int j = 0; j < _matrix.GetLength(1); j++)
            {
                output[i] += _matrix[i, j] * input[j];
            }
        }

        return new Vector(output);
    }

    public void Backward(Vector error, Vector input, float alpha)
    {
        for(int i = 0; i < _matrix.GetLength(0); i++)
        {
            for(int j = 0; j < _matrix.GetLength(1); j++)
            {
                _matrix[i, j] -= alpha * error[i] / input[j];
            }
        }
    }
}
