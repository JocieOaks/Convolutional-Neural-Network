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

    [JsonProperty] UnionMap _map;

    public FeatureMap(int width, int length)
    {
        Width = width;
        Length = length;

        _map = new UnionMap(width * length);
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

    [JsonIgnore] public int Area => Length * Width;

    [JsonIgnore] public int Length { get; }

    [JsonIgnore] public int Width { get; }

    public Color this[int x, int y]
    {
        get => _map[y * Width + x];
        set => _map[y * Width + x] = value;
    }
    public MemoryBuffer1D<Color, Stride1D.Dense> Allocate(Accelerator accelerator)
    {
        MemoryBuffer1D<Color, Stride1D.Dense> buffer = accelerator.Allocate1D<Color>(Area);
        buffer.AsArrayView<Color>(0, Area).CopyFromCPU(_map.ColorsSpanReadonly(Area));
        return buffer;
    }

    public MemoryBuffer1D<Color, Stride1D.Dense> AllocateEmpty(Accelerator accelerator)
    {
        return accelerator.Allocate1D<Color>(Area);
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
        buffer.AsArrayView<Color>(0, Area).CopyToCPU(_map.ColorsSpan(Area));
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
        buffer.CopyToCPU(_map.Floats);
    }

    [StructLayout(LayoutKind.Explicit)]
    private struct UnionMap
    {
        [FieldOffset(0)] Color[] _colors;
        [FieldOffset(0)] float[] _floats;

        public float[] Floats => _floats;
        public Color[] Colors => _colors;
        public ReadOnlySpan<Color> ColorsSpanReadonly(int length) => new ReadOnlySpan<Color>(_colors, 0, length);
        public Span<Color> ColorsSpan(int length) => new Span<Color>(_colors, 0, length);

        public Color this[int index]
        {
            get => _colors[index];
            set => _colors[index] = value;
        }

        public UnionMap(int length)
        {
            _floats = new float[length * 3];
        }
    }
}