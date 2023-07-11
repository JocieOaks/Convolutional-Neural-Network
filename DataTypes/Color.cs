using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Runtime.InteropServices;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="Color"/> struct contains color data for <see cref="FeatureMap"/>s. However, <see cref="Color"/> does not represent
    /// real world colors, but stores data related to the RGB channels of a <see cref="FeatureMap"/>. Values can be negative or greated than 1.
    /// Creating images requires normalizing the values to a representation that is within the range of <see cref="System.Drawing.Color"/>.
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential, Size = 12)]
    public readonly struct Color
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Color"/> struct.
        /// </summary>
        /// <param name="r">The red channel value.</param>
        /// <param name="g">The green channel value.</param>
        /// <param name="b">The blue channel value.</param>
        [JsonConstructor]
        public Color(float r, float g, float b)
        {
            _1r = r;
            _2g = g;
            _3b = b;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Color"/> struct with default values of 0.
        /// </summary>
        public Color()
        {
            _1r = 0;
            _2g = 0;
            _3b = 0;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Color"/> struct with all three RGB values sharing the same value.
        /// </summary>
        /// <param name="unit">The value to set RGB to.</param>
        public Color(float unit)
        {
            _1r = unit;
            _2g = unit;
            _3b = unit;
        }

        public static Color Zero => new(0);

        public static Color One => new(1);

        private readonly float _1r;
        private readonly float _2g;
        private readonly float _3b;

        /// <value>The <see cref="Color"/>'s blue channel value.</value>
        public float B => _3b;

        /// <value>The <see cref="Color"/>'s green channel value.</value>
        public float G => _2g;

        /// <value>The <see cref="Color"/>'s red channel value.</value>
        public float R => _1r;

        /// <summary>
        /// Gives the RGB value at the given index. Used in <see cref="ILGPU"/> kernals where the color channel is one of the dimensions being indexed.
        /// </summary>
        /// <param name="index">The index of the value.</param>
        /// <returns>Returns the value corresponding to the index given. Defaults to blue for any value besides 0, 1 or 2.</returns>
        /// Note: Using the modulus operator in an external function creates an error when used in an <see cref="ILGPU"/> kernal,
        /// so the value defaults to blue.
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

        /// <summary>
        /// Calculates the dot product between two <see cref="Color"/>s represented as vectors.
        /// </summary>
        /// <param name="color1">The first <see cref="Color"/>.</param>
        /// <param name="color2">The second <see cref="Color"/>.</param>
        /// <returns>Returns the dot product of the colors.</returns>
        public static float Dot(Color color1, Color color2)
        {
            return color1.R * color2.R + color1.G * color2.G + color1.B * color2.B;
        }

        /// <summary>
        /// Explicit conversion between an <see cref="ILGPU"/> <see cref="MemoryBuffer1D{T, TStride}"/> of three floats.
        /// The buffer is expected to exclusively be of three floats.
        /// </summary>
        /// <param name="array">The buffer representation of the <see cref="Color"/>.</param>
        public static explicit operator Color(MemoryBuffer1D<float, Stride1D.Dense> array)
        {
            float[] values = new float[3];
            array.CopyToCPU(values);
            return new Color(values[0], values[1], values[2]);
        }

        /// <summary>
        /// Explicit conversion from a <see cref="System.Drawing.Color"/> to <see cref="Color"/>.
        /// </summary>
        /// <param name="color">The <see cref="System.Drawing.Color"/> being converted into a <see cref="Color"/>.</param>
        public static explicit operator Color(System.Drawing.Color color)
        {
            return new Color(color.R / 255f, color.G / 255f, color.B / 255f) * (color.A / 255f);
        }

        /// <summary>
        /// Explicit conversion conversions from <see cref="Color"/> to <see cref="System.Drawing.Color"/>.
        /// Constrains RGB values to integers between 0 and 255.
        /// </summary>
        /// <param name="color">The <see cref="Color"/> being converted.</param>
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

        public static bool operator ==(Color color1, Color color2)
        {
            for(int i = 0; i <3; i++)
            {
                if (MathF.Abs(color1[i] - color2[i]) > MathF.Max(1e-3f, 1e-3f * Math.Abs(color1[i] + color2[i])))
                    return false;
            }
            return true;
        }

        public static bool operator !=(Color color1, Color color2)
        {
            for (int i = 0; i < 3; i++)
            {
                if (MathF.Abs(color1[i] - color2[i]) > MathF.Max(1e-3f, 1e-3f * Math.Abs(color1[i] + color2[i])))
                    return true;
            }
            return false;
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

        /// <summary>
        /// Raises the given <see cref="Color"/> to a given power, exponentiating each channel value individually.
        /// </summary>
        /// <param name="color">The <see cref="Color"/>.</param>
        /// <param name="power">The power the <see cref="Color"/> is being raised to.</param>
        /// <returns>Returns <paramref name="color"/> raised to <paramref name="power"/>.</returns>
        public static Color Pow(Color color, float power)
        {
            return new Color(MathF.Pow(color.R, power), MathF.Pow(color.G, power), MathF.Pow(color.B, power));
        }

        /// <summary>
        /// Generates a new random <see cref="Color"/> whose values belong to a gaussian distribution with the given mean and standard deviation.
        /// </summary>
        /// <param name="mean">The mean of the distribtion.</param>
        /// <param name="stdDev">The standard deviation.</param>
        /// <returns>Returns a new random <see cref="Color"/> from the distribtion.</returns>
        public static Color RandomGauss(float mean, float stdDev)
        {
            return new Color(ConvolutionalNeuralNetwork.Utility.RandomGauss(mean, stdDev), ConvolutionalNeuralNetwork.Utility.RandomGauss(mean, stdDev), ConvolutionalNeuralNetwork.Utility.RandomGauss(mean, stdDev));
        }

        /// <summary>
        /// Restricts the <see cref="Color"/>s values to have an absolute value less than or equal to the given value. Used for gradient clipping.
        /// </summary>
        /// <param name="val">The maximum the absolute value of the <see cref="Color"/>'s values can be.</param>
        /// <returns>Returns a new <see cref="Color"/> restrict to within the range of <paramref name="val"/>.</returns>
        public Color Clamp(float val)
        {
            return new Color(R > val ? val : R < -val ? -val : R, G > val ? val : G < -val ? -val : G, B > val ? val : B < -val ? -val : B);
        }

        /// <summary>
        /// Performs the Rectified Linear Unit activation function on each value of the <see cref="Color"/>.
        /// (Currently modified to use Leaky ReLU instead of ReLU).
        /// </summary>
        /// <returns>Returns the <see cref="Color"/> after going through activation.</returns>
        public Color ReLU()
        {
            return new Color(R < 0 ? 0.01f * R : R, G < 0 ? 0.01f * G : G, B < 0 ? 0.01f * B : B);
        }

        /// <summary>
        /// Gives the <see cref="Color"/> coefficients for backpropogating through the activation function.
        /// (Currently modified to use Leaky ReLU instead of ReLU).
        /// </summary>
        /// <returns>Returns the coefficients multiplied by the activation function when performing activation.</returns>
        public Color ReLUPropagation()
        {
            return new Color(R < 0 ? 0.01f : 1, G < 0 ? 0.01f : 1, B < 0 ? 0.01f : 1);
        }
    }
}