// See https://aka.ms/new-console-template for more information
using Newtonsoft.Json;
using System.Drawing.Drawing2D;
using System.Xml;

[Serializable]
public class VectorizationLayer
{
    [JsonProperty]
    readonly Color[,] _matrix;

    public VectorizationLayer(int vectorDimensions, int kernalNum)
    {
        _matrix = new Color[vectorDimensions, kernalNum];
        for(int i = 0; i < vectorDimensions; i++)
        {
            for(int j = 0; j < kernalNum; j++)
            {
                _matrix[i, j] = new Color().Random();
            }
        }
    }

    public Vector Forward(Color[][,] input) 
    { 
        Color[] vector = new Color[input.Length];

        for(int i = 0; i < input.Length; i++)
        {
            for(int j = 0; j < input[i].GetLength(0); j++)
            {
                for(int k = 0; k < input[i].GetLength(1); k++)
                {
                    vector[i] = vector[i].Add(input[i][j, k]);
                }
            }
            vector[i] = vector[i].Multiply(1f / (input[i].GetLength(0) * input[i].GetLength(1)));
        }

        Vector output = new(_matrix.GetLength(0));
        for (int i = 0; i < _matrix.GetLength(0); i++)
        {
            for (int j = 0; j < _matrix.GetLength(1); j++)
            {
                output[i] += _matrix[i, j].Multiply(vector[j]).Magnitude;
            }
        }

        return output;
    }

    public Color[][,] Backwards(Vector dL_dI, Vector output, Color[][,] input, float alpha)
    {
        Color[][,] dL_dP = new Color[input.Length][,];
        int x = input[0].GetLength(0);
        int y = input[0].GetLength(1);
        float _xy = 1f / (x * y);

        Color[] dL_dPV = new Color[_matrix.GetLength(1)];
        for (int i = 0; i < _matrix.GetLength(0); i++)
        {
            for (int j = 0; j < _matrix.GetLength(1); j++)
            {
                dL_dPV[j] = dL_dPV[j].Add(_matrix[i, j].Multiply(dL_dI[i] * _xy));
            }
        }

        for(int i = 0; i < input.Length; i++)
        {
            dL_dP[i] = new Color[x, y];
            for(int j = 0; j < x; j++)
            {
                for(int k = 0; k < y; k++)
                {
                    dL_dP[i][j, k] = dL_dPV[i];
                }
            }
        }


        for (int i = 0; i < _matrix.GetLength(0); i++)
        {
            for (int j = 0; j < _matrix.GetLength(1); j++)
            {
                _matrix[i, j] = _matrix[i,j].Subtract(dL_dPV[j].Multiply(alpha * dL_dI[i]));
            }
        }

        return dL_dP;
    }
}
