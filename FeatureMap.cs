using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

[System.Serializable]
public class FeatureMap
{

    [JsonProperty] Color[] _map;

    public FeatureMap(int width, int length)
    {
        Width = width;
        Length = length;

        _map = new Color[width * length];
    }

    public FeatureMap(int width, int length, Color initial) : this(width, length)
    {
        Width = width;
        Length = length;

        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < width; j++)
            {
                _map[i * Width + j] = initial;
            }
        }
    }

    [JsonIgnore] public int Area => _map.Length;

    [JsonIgnore] public int FloatLength => _map.Length * 3;

    [JsonIgnore] public int Length { get; }

    [JsonIgnore] public int Width { get; }

    public Color this[int x, int y]
    {
        get => _map[y * Width + x];
        set => _map[y * Width + x] = value;
    }
    public MemoryBuffer1D<Color, Stride1D.Dense> Allocate(Accelerator accelerator)
    {
        return accelerator.Allocate1D(_map);
    }

    public MemoryBuffer1D<Color, Stride1D.Dense> AllocateEmpty(Accelerator accelerator)
    {
        return accelerator.Allocate1D<Color>(Area);
    }

    public MemoryBuffer1D<float, Stride1D.Dense> AllocateFloat(Accelerator accelerator)
    {
        return accelerator.Allocate1D<float>(FloatLength);
    }

    public Color Average()
    {
        return Sum() / Area;
    }

    public float AverageMagnitude()
    {
        return SumMagnitude() / Area;
    }

    public void CopyFromBuffer(MemoryBuffer1D<Color, Stride1D.Dense> buffer)
    {
        buffer.CopyToCPU(_map);
    }

    public Color Sum()
    {
        Color color = new();
        for(int i = 0; i < Length; i++)
        {
            for(int j = 0; j < Width; j++)
            {
                color += _map[i * Width + j];
            }
        }

        return color;
    }
    public float SumMagnitude()
    {
        float sum = 0;
        for (int i = 0; i < Length; i++)
        {
            for (int j = 0; j < Width; j++)
            {
                sum += _map[i * Width + j].Magnitude;
            }
        }

        return sum;
    }

    public void CopyFromBuffer(MemoryBuffer1D<float, Stride1D.Dense> buffer)
    {
        unsafe
        {
            fixed (void* ptr = &_map[0])
            {
                Span<float> span = new Span<float>(ptr, Area * 3);
                buffer.AsArrayView<float>(0, Area * 3).CopyToCPU(span);
            } 
        }
    }
}