﻿using System.Drawing;
using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Example
{
    /// <summary>
    /// The <see cref="ImageChannels"/> enum is used to indicate the number of color channels used by images generated by the network.
    /// </summary>
    public enum ImageChannels
    {
        /// <value>Used for greyscale images.</value>
        Greyscale = 1,
        /// <value>Used for RGB images.</value>
        RGB = 3,
        /// <value>Used for RGBA images.</value>
        RGBA = 4
    }

    /// <summary>
    /// Utility functions for working with <see cref="Tensor"/>s. Bitmap conversion requires a Windows operating system.
    /// </summary>
    public static class TensorUtility
    {
        /// <summary>
        /// Creates a new <see cref="Tensor"/> from a given <see cref="Bitmap"/>.
        /// </summary>
        /// <param name="bitmap">The <see cref="Bitmap"/> being converted.</param>
        /// <param name="channels">The <see cref="ImageChannels"/> to use when converting the <see cref="Bitmap"/> into a <see cref="Tensor"/>.</param>
        /// <param name="width">The width of the converted <see cref="Tensor"/>. Defaults to <see cref="Bitmap"/>'s width.</param>
        /// <param name="length">The length of the converted <see cref="Tensor"/>. Defaults to <see cref="Bitmap"/>'s length.</param>
        /// <returns>Returns the <see cref="Tensor"/> representation of the <see cref="Bitmap"/>.</returns>
        public static Tensor BitmapToTensor(Bitmap bitmap, ImageChannels channels, int width = -1, int length = -1)
        {
            if (!OperatingSystem.IsWindows()) return null;

            if (width == -1 || length == -1)
            {
                width = bitmap.Width;
                length = bitmap.Height;
            }

            int paddingX = (width - bitmap.Width) / 2;
            int paddingY = (length - bitmap.Height) / 2;

            Tensor tensor = new(new TensorShape(width, length, (int)channels));
            for (int i = 0; i < tensor.Dimensions; i++)
            {
                for (int y = 0; y < bitmap.Height; y++)
                {
                    for (int x = 0; x < bitmap.Width; x++)
                    {
                        tensor[paddingX + x, paddingY + bitmap.Height - y - 1, i] = i switch
                        {
                            0 => (bitmap.GetPixel(x, y).R - 127.5f) / 127.5f,
                            1 => (bitmap.GetPixel(x, y).G - 127.5f) / 127.5f,
                            2 => (bitmap.GetPixel(x, y).B - 127.5f) / 127.5f,
                            _ => (bitmap.GetPixel(x, y).A - 127.5f) / 127.5f,
                        };
                    }
                }
            }

            return tensor;
        }

        /// <summary>
        /// Generates a new randomized <see cref="Tensor"/> of the given size.
        /// </summary>
        public static Tensor RandomTensor(int width, int length, int dimensions)
        {
            Tensor map = new(new TensorShape(width, length, dimensions));
            for (int dimension = 0; dimension < dimensions; dimension++)
            {
                for (int y = 0; y < length; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        map[x, y, dimension] = Math.Clamp(Utility.RandomGauss(0, 0.4f), -1, 1);
                    }
                }
            }

            return map;
        }

        /// <summary>
        /// Creates a <see cref="Bitmap"/> representation of the <see cref="Tensor"/>.
        /// </summary>
        /// <returns>Returns the <see cref="Tensor"/> as a <see cref="Bitmap"/>.</returns>
        public static Bitmap TensorToBitmap(Tensor tensor)
        {
            if (!OperatingSystem.IsWindows()) return null;

            Bitmap bitmap = new(tensor.Width, tensor.Length);

            for (int y = 0; y < bitmap.Height; y++)
            {
                for (int x = 0; x < bitmap.Width; x++)
                {
                    if (tensor.Dimensions == 1)
                    {
                        bitmap.SetPixel(x, bitmap.Height - y - 1, Color.FromArgb(Math.Clamp((int)(tensor[x, y, 0] * 255), 0, 255), Color.White));
                    }
                    else if (tensor.Dimensions == 3)
                    {
                        bitmap.SetPixel(x, bitmap.Height - y - 1, Color.FromArgb(
                            Math.Clamp((int)(tensor[x, y, 0] * 127.5 + 127.5), 0, 255),
                            Math.Clamp((int)(tensor[x, y, 1] * 127.5 + 127.5), 0, 255),
                            Math.Clamp((int)(tensor[x, y, 2] * 127.5 + 127.5), 0, 255))
                        );
                    }
                    else
                    {
                        bitmap.SetPixel(x, bitmap.Height - y - 1, Color.FromArgb(
                            Math.Clamp((int)(tensor[x, y, 3] * 127.5 + 127.5), 0, 255),
                            Math.Clamp((int)(tensor[x, y, 0] * 127.5 + 127.5), 0, 255),
                            Math.Clamp((int)(tensor[x, y, 1] * 127.5 + 127.5), 0, 255),
                            Math.Clamp((int)(tensor[x, y, 2] * 127.5 + 127.5), 0, 255))
                        );
                    }
                }
            }

            return bitmap;
        }
    }
}
