using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

public class FeatureMap
{

    readonly Color[,] _map;

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

    public int Width => _map.GetLength(0);
    public int Length => _map.GetLength(1);

    public int Area => Width * Length;

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
}