using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

public struct Vector
{
    readonly float[] _values;

    public float this[int index]
    {
        get { return _values[index]; }
    }

    public Vector(float[] values)
    {
        _values = values;
    }

    public Vector(int length)
    {
        _values = new float[length];
    }

    public int Length => _values.Length;

    public static float Cos(Vector v1, Vector v2)
    {
        if (v1.Length != v2.Length)
        {
            throw new ArgumentException("Vector's not the same length.");
        }

        return Dot(v1, v2) / (Magnitude(v1) * Magnitude(v2));
    }

    public static float Dot(Vector v1, Vector v2)
    {
        if (v1.Length != v2.Length)
        {
            throw new ArgumentException("Vector's not the same length.");
        }

        float dot = 0;

        for (int i = 0; i < v1.Length; i++)
        {
            dot += v1[i] * v2[i];
        }

        return dot;
    }

    public static float Magnitude(Vector vector)
    {
        float sum = 0;
        for (int i = 0; i < vector.Length; i++)
        {
            sum += vector[i] * vector[i];
        }

        return MathF.Sqrt(sum);
    }

    public static Vector operator +(Vector v1, Vector v2)
    {
        if (v1.Length != v2.Length)
        {
            throw new ArgumentException("Vector's not the same length.");
        }

        float[] values = new float[v1.Length];

        for (int i = 0; i < v1.Length; i++)
        {
            values[i] = v1[i] + v2[i];
        }
        return new Vector(values);
    }

    public static Vector operator -(Vector v1, Vector v2)
    {
        if (v1.Length != v2.Length)
        {
            throw new ArgumentException("Vector's not the same length.");
        }

        float[] values = new float[v1.Length];

        for (int i = 0; i < v1.Length; i++)
        {
            values[i] = v1[i] - v2[i];
        }
        return new Vector(values);
    }
}

