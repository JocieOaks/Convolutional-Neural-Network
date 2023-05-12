using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

public abstract class Layer
{
    [JsonProperty]
    protected int _kernalSize;
    [JsonProperty]
    protected int _stride;
    [JsonProperty]
    protected int _dimensions;

    public Layer(int dimensions, int kernalSize, int stride)
    {
        _kernalSize = kernalSize;
        _stride = stride;
        _dimensions = dimensions;
    }

    protected void Pad(FeatureMap[] input)
    {
        int paddingX = (input[0].Width - _kernalSize) % _stride;
        int paddingY = (input[0].Length - _kernalSize) % _stride;
        if (paddingX == 0 && paddingY == 0)
            return;

        int halfX = paddingX / 2;
        int halfY = paddingY / 2;
        for (int k = 0; k < input.Length; k++)
        {
            FeatureMap padded = new FeatureMap(input[k].Width + paddingX, input[k].Length + paddingY);
            for (int i = 0; i < input[k].Width; i++)
            {
                for (int j = 0; j < input[k].Length; j++)
                {
                    padded[i + halfX, j + halfY] = input[k][i, j];
                }
            }
            input[k] = padded;
        }
        return;
    }

    public abstract FeatureMap[][] Forward(FeatureMap[][] input);

    public abstract FeatureMap[][] Backwards(FeatureMap[][] dL_dP, FeatureMap[][] input, float alpha);
}

