// See https://aka.ms/new-console-template for more information
using Newtonsoft.Json;

[Serializable]
public class Transformer
{
    [JsonProperty]
    readonly float[,] _matrix;

    public Transformer(int inputs, int outputs)
    {
        _matrix = new float[outputs, inputs];
        for(int i = 0; i < outputs; i++)
        {
            for(int j = 0; j < inputs; j++)
            {
                _matrix[i, j] = (float)CLIP.Random.NextDouble();
            }
        }
    }

    public Vector Forward(int[] input)
    {

        return _matrix * new Vector(input);
    }

    public void Backwards(Vector dL_dT, int[] input, float alpha)
    {
        for(int i = 0; i < _matrix.GetLength(0); i++)
        {
            for(int j = 0; j < _matrix.GetLength(1); j++)
            {
                _matrix[i, j] -= alpha * dL_dT[i] * input[j];
            }
        }
    }
}
