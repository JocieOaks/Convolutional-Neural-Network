// See https://aka.ms/new-console-template for more information

using Newtonsoft.Json;
using System.Data.Common;
using System.Diagnostics.Contracts;

[Serializable]
public readonly struct Color
{
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

    public float B { get; }
    public float G { get; }
    public float Magnitude => MathF.Sqrt(R * R + G * G + B * B);

    public float SquareMagnitude => R * R + G * G + B * B;
    public float R { get; }
    public static Color operator -(Color color1, Color color2)
    {
        return new Color(color1.R - color2.R, color1.G - color2.G, color1.B - color2.B);
    }

    public static Color operator -(Color color)
    {
        return new Color(-color.R, -color.G, -color.B);
    }

    public static Color operator *(Color color1, Color color2)
    {
        return new Color(color1.R * color2.R, color1.G * color2.G, color1.B * color2.B);
    }

    public static Color operator *(Color color, float multiple)
    {
        return new Color(color.R * multiple, color.G * multiple, color.B * multiple);
    }

    public static Color operator *(float multiple, Color color)
    {
        return new Color(color.R * multiple, color.G * multiple, color.B * multiple);
    }

    public static Color operator /(Color color1, Color color2)
    {
        return new Color(color1.R / color2.R, color1.G / color2.G, color1.B / color2.B);
    }

    public static Color operator /(Color color, int divisor)
    {
        return new Color(color.R / divisor, color.G / divisor, color.B / divisor);
    }

    public static Color operator +(Color color1, Color color2)
    {
        return new Color(color1.R + color2.R, color1.G + color2.G, color1.B + color2.B);
    }

    public static Color Pow(Color color, float power)
    {
        return new Color(MathF.Pow(color.R, power), MathF.Pow(color.G, power), MathF.Pow(color.B, power));
    }

    public static Color Random(float range)
    {

        return new Color((float)(CLIP.Random.NextDouble() - 0.5f) * 2 * range, (float)(CLIP.Random.NextDouble() - 0.5f) * 2 * range, (float)(CLIP.Random.NextDouble() - 0.5f) * 2 * range);
    }

    public Color ReLU()
    {
        return new Color(R < 0 ? 0 : R, G < 0 ? 0 : G, B < 0 ? 0 : B);
    }

    public override string ToString()
    {
        return "R: " + MathF.Round(R, 2) + " G: " + MathF.Round(G, 2) + " B: " + MathF.Round(B, 2);
    }

    public Color ReLUPropogation()
    {
        return new Color(R == 0 ? 0 : 1, G == 0 ? 0 : 1, B == 0 ? 0 : 1);
    }

    public Color Clamp(float val)
    {
        return new Color(R > val ? val : R < -val ? -val : R, G > val ? val : G < -val ? -val : G, B > val ? val : B < -val ? -val : B);
    }

    public static float Dot(Color color1, Color color2)
    {
        return color1.R * color2.R + color1.G * color2.G + color1.B * color2.B;
    }
}
