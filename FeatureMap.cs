using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

[System.Serializable]
public class FeatureMap
{

    [JsonProperty] readonly Color[,] _map;

    public FeatureMap(int width, int length)
    {
        _map = new Color[width, length];
    }

    public Color this[int x, int y]
    {
        get => _map[x, y];
        set => _map[x, y] = value;
    }

    public FeatureMap(int width, int length, Color initial) : this(width, length)
    { 
        for(int i = 0; i < width; i++)
        {
            for(int j = 0; j < length; j++)
            {
                _map[i, j] = initial;
            }
        }
    }

    [JsonIgnore] public int Width => _map.GetLength(0);
    [JsonIgnore] public int Length => _map.GetLength(1);

    [JsonIgnore] public int Area => Width * Length;

    public Color Sum()
    {
        Color color = new();
        for(int i = 0; i < Width; i++)
        {
            for(int j = 0; j < Length; j++)
            {
                color += _map[i, j];
            }
        }

        return color;
    }

    public Color Average()
    {
        return Sum() / Area;
    }

    public float SumMagnitude()
    {
        float sum = 0;
        for (int i = 0; i < Width; i++)
        {
            for (int j = 0; j < Length; j++)
            {
                sum += (float)_map[i, j].Magnitude;
            }
        }

        return sum;
    }

    public float AverageMagnitude()
    {
        return SumMagnitude() / Area;
    }
}