using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Runtime.InteropServices;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    [Serializable]
    [StructLayout(LayoutKind.Sequential, Size = 12)]
    public readonly struct Color
    {
        [JsonConstructor]
        public Color(float r, float g, float b)
        {
            _1r = r;
            _2g = g;
            _3b = b;
        }

        public Color()
        {
            _1r = 0;
            _2g = 0;
            _3b = 0;
        }

        public Color(float unit)
        {
            _1r = unit;
            _2g = unit;
            _3b = unit;
        }

        private readonly float _1r;
        private readonly float _2g;
        private readonly float _3b;

        public float B => _3b;
        public float G => _2g;

        [JsonIgnore] public float Magnitude => MathF.Sqrt(R * R + G * G + B * B);

        public float R => _1r;

        public float this[int index]
        {
            get
            {
                return index switch
                {
                    0 => R,
                    1 => G,
                    _ => B
                };
            }
        }

        [JsonIgnore] public float SquareMagnitude => R * R + G * G + B * B;

        public static float Dot(Color color1, Color color2)
        {
            return color1.R * color2.R + color1.G * color2.G + color1.B * color2.B;
        }

        public static explicit operator Color(MemoryBuffer1D<float, Stride1D.Dense> array)
        {
            float[] values = new float[3];
            array.CopyToCPU(values);
            return new Color(values[0], values[1], values[2]);
        }

        public static explicit operator Color(System.Drawing.Color color)
        {
            int inverseAlpha = 255 - color.A;
            return new Color((color.R + inverseAlpha) / 255f, (color.G + inverseAlpha) / 255f, (color.B + inverseAlpha) / 255f);
        }

        public static explicit operator System.Drawing.Color(Color color)
        {
            int r = Math.Max(Math.Min((int)color.R, 255), 0);
            int g = Math.Max(Math.Min((int)color.G, 255), 0);
            int b = Math.Max(Math.Min((int)color.B, 255), 0);
            return System.Drawing.Color.FromArgb(r, g, b);
        }

        public static Color operator -(Color color1, Color color2)
        {
            return new Color(color1.R - color2.R, color1.G - color2.G, color1.B - color2.B);
        }

        public static Color operator -(Color color)
        {
            return new Color(-color.R, -color.G, -color.B);
        }

        public static Color operator -(Color color, ArrayView<float> array)
        {
            return new Color(color.R - array[0], color.G - array[1], color.B - array[2]);
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

        public static ArrayView<float> operator +(ArrayView<float> array, Color color)
        {
            Atomic.Add(ref array[0], color.R);
            Atomic.Add(ref array[1], color.G);
            Atomic.Add(ref array[2], color.B);
            return array;
        }

        public static Color Pow(Color color, float power)
        {
            return new Color(MathF.Pow(color.R, power), MathF.Pow(color.G, power), MathF.Pow(color.B, power));
        }

        public static Color RandomGauss(float mean, float stdDev)
        {
            return new Color(ConvolutionalNeuralNetwork.Utility.RandomGauss(mean, stdDev), ConvolutionalNeuralNetwork.Utility.RandomGauss(mean, stdDev), ConvolutionalNeuralNetwork.Utility.RandomGauss(mean, stdDev));
        }

        public Color Clamp(float val)
        {
            return new Color(R > val ? val : R < -val ? -val : R, G > val ? val : G < -val ? -val : G, B > val ? val : B < -val ? -val : B);
        }

        public Color ReLU()
        {
            return new Color(R < 0 ? 0.1f * R : R, G < 0 ? 0.1f * G : G, B < 0 ? 0.1f * B : B);
        }

        public Color ReLUPropogation()
        {
            return new Color(R < 0 ? 0.1f : 1, G < 0 ? 0.1f : 1, B < 0 ? 0.1f : 1);
        }

        public override string ToString()
        {
            return "R: " + MathF.Round(R, 2) + " G: " + MathF.Round(G, 2) + " B: " + MathF.Round(B, 2);
        }

        public float[] ToArray()
        {
            return new float[] { R, G, B };
        }
    }
}