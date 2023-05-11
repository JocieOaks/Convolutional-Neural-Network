// See https://aka.ms/new-console-template for more information

using Newtonsoft.Json;

[Serializable]
public readonly struct Color : IDot<Color>
{
    public float R { get; }
    public float G { get; }
    public float B { get; }
    

    public Color Multiply(Color other)
    {
        return new Color(R * other.R, G *  other.G, B * other.B);
    }

    public Color Random()
    {
        return new Color((float)CLIP.Random.NextDouble(), (float)CLIP.Random.NextDouble(), (float)CLIP.Random.NextDouble());
    }

    public Color Add(Color other)
    {
        return new Color(R + other.R, G + other.G, B + other.B);
    }

    public Color Multiply(float multiple)
    {
        return new Color(R * multiple, G * multiple, B * multiple);
    }

    public Color Divide(Color other)
    {
        return new Color(R / other.R, G / other.G, B / other.B);
    }

    public Color ReLU()
    {
        return new Color(R < 0 ? 0 : R, G < 0 ? 0 : G, B < 0 ? 0 : B);
    }

    public Color Subtract(Color other)
    {
        return new Color(R - other.R, G - other.G, B - other.B);
    }

    public Color Pow(float power)
    {
        return new Color(MathF.Pow(R, power), MathF.Pow(G, power), MathF.Pow(B, power));
    }

    public Color Add(float value)
    {
        return new Color(R + value, G + value, B + value);
    }

    [JsonConstructor]
    public Color(float r, float g, float b)
    {
        R = r;
        G = g;
        B = b;
    }

    public Color()
    {
        R = 0;
        G = 0;
        B = 0;
    }

    public Color(System.Drawing.Color color)
    {
        int inverseAlpha = 255 - color.A;
        R = (color.R + inverseAlpha) / 255f;
        G = (color.G + inverseAlpha) / 255f;
        B = (color.B + inverseAlpha) / 255f;
    }

    public float Magnitude => MathF.Sqrt(R * R + G * G + B * B);
}
