using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Drawing;
using System.Reflection.Metadata.Ecma335;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="FeatureMap"/> class contains color data for the input and ouputs of layers and their gradients.
    /// (In some cases <see cref="FeatureMap"/> is used as an n x m x 3 tensor, because that is what it fundamentally is. However, this
    /// can be confusing, so it may be better to create a Tensor class of which <see cref="FeatureMap"/> is a child.)
    /// </summary>
    [Serializable]
    public class FeatureMap : Tensor
    {
        private readonly static float s_colorMean = 0;
        private readonly static float s_colorDeviation = 0.5f;
        private readonly static float s_colorMeanR = 0.059f;
        private readonly static float s_colorDeviationR = 0.158f;
        private readonly static float s_colorMeanG = 0.052f;
        private readonly static float s_colorDeviationG = 0.143f;
        private readonly static float s_colorMeanB = 0.047f;
        private readonly static float s_colorDeviationB = 0.126f;

        /// <summary>
        /// Initializes a new <see cref="FeatureMap"/> with the given dimensions.
        /// </summary>
        /// <param name="width">The width of the <see cref="FeatureMap"/>.</param>
        /// <param name="length">The length of the <see cref="FeatureMap"/>.</param>
        public FeatureMap(int width, int length) : base(width, length) { }

        public FeatureMap(Shape shape) : base(shape) { }

        public FeatureMap(int width, int length, float value) : base(width, length)
        {
            for(int i = 0; i < Area; i++)
            {
                _tensor[i] = value;
            }
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor]
        private FeatureMap()
        {
        }

        private static void PixelPallet(FeatureMap[] source, FeatureMap[] target)
        {
            List<Vector> pallet = new();
            Vector color = new(source.Length);
            for (int y = 0; y < source[0].Length; y++)
            {
                for (int x = 0; x < source[0].Width; x++)
                {
                    for (int i = 0; i < source.Length; i++)
                    {
                        color[i] = source[i][x, y];
                    }

                    if (!pallet.Contains(color))
                    {
                        pallet.Add(color);
                        color = new Vector(source.Length);
                    }
                }
            }

            for (int y = 0; y < target[0].Length; y++)
            {
                for (int x = 0; x < target[0].Width; x++)
                {
                    for (int i = 0; i < source.Length; i++)
                    {
                        color[i] = target[i][x, y];
                    }
                    Vector correction = pallet.MinBy(x => Vector.DistanceSquared(x, color));

                    for (int i = 0; i < target.Length; i++)
                    {
                        target[i][x, y] = correction[i];
                    }
                }
            }
        }

        /// <summary>
        /// Creates a new <see cref="FeatureMap"/> from a given <see cref="Bitmap"/>.
        /// </summary>
        /// <param name="bitmap">The <see cref="Bitmap"/> being converted.</param>
        /// <returns>Returns the <see cref="FeatureMap"/> representation of the <see cref="Bitmap"/>.</returns>
        public static FeatureMap[] FromBitmap(Bitmap bitmap, int channels, int width = -1, int length = -1)
        {
            if (OperatingSystem.IsWindows())
            {
                if(width == -1 || length == -1)
                {
                    width = bitmap.Width;
                    length = bitmap.Height;
                }

                int paddingX = (width - bitmap.Width) / 2;
                int paddingY = (length - bitmap.Height) / 2;

                FeatureMap[] maps = new FeatureMap[channels];
                System.Drawing.Color background = bitmap.GetPixel(0, 0);
                for (int i = 0; i < maps.Length; i++)
                {
                    maps[i] = new(width, length);
                    for(int y = 0; y < length; y++)
                    {
                        for(int x = 0; x < width; x++)
                        {
                            maps[i][x, y] = i switch {
                                0 => (background.R - 127.5f) / 127.5f,
                                1 => (background.G - 127.5f) / 127.5f,
                                2 => (background.B - 127.5f) / 127.5f,
                                _ => (background.A - 127.5f) / 127.5f,
                            };
                        }
                    }

                    for (int y = 0; y < bitmap.Height; y++)
                    {
                        for (int x = 0; x < bitmap.Width; x++)
                        {
                            maps[i][paddingX + x, paddingY + bitmap.Height - y - 1] = i switch
                            {
                                0 => (bitmap.GetPixel(x, y).R - 127.5f) / 127.5f,
                                1 => (bitmap.GetPixel(x, y).G - 127.5f) / 127.5f,
                                2 => (bitmap.GetPixel(x, y).B - 127.5f) / 127.5f,
                                _ => (bitmap.GetPixel(x, y).A - 127.5f) / 127.5f,
                            };
                        }
                    }
                }

                return maps;
            }
            return null;
        }

        /// <summary>
        /// Generates a new <see cref="FeatureMap"/> of the given dimensions with randomized values. Used for creating starting noise
        /// for a <see cref="Networks.Generator"/>.
        /// </summary>
        /// <param name="width"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        public static FeatureMap Random(int width, int length)
        {
            FeatureMap map = new(width, length);
            for (int y = 0; y < length; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    map[x, y] = Math.Clamp(Utility.RandomGauss(0.5f, 0.2f), -1, 1);
                }
            }
            return map;
        }

        /// <summary>
        /// Calculates the average <see cref="Color"/> of the <see cref="FeatureMap"/>.
        /// </summary>
        /// <returns>Returns the <see cref="Color"/>.</returns>
        public float Average()
        {
            return Sum() / Area;
        }

        /// <summary>
        /// Creates a <see cref="Bitmap"/> representation of the <see cref="FeatureMap"/>.
        /// The map is normalized to fit in the range of <see cref="System.Drawing.Color"/>.
        /// </summary>
        /// <param name="setNormalized">If true, the <see cref="FeatureMap"/> will be set to the normalized values.</param>
        /// <returns>Returns the <see cref="FeatureMap"/> as a <see cref="Bitmap"/>.</returns>
        public static Bitmap ConstructBitmap(FeatureMap[,] maps, int index)
        {
            if (OperatingSystem.IsWindows())
            {
                Bitmap bitmap = new(maps[index, 0].Width, maps[index, 0].Length);

                float[][] normalizedMaps = new float[maps.GetLength(1)][];

                for (int i = 0; i < maps.GetLength(1); i++)
                {
                    normalizedMaps[i] = i switch
                    {
                        0 => Normalize(maps[index, i], s_colorMeanR, s_colorDeviationR),
                        1 => Normalize(maps[index, i], s_colorMeanG, s_colorDeviationG),
                        2 => Normalize(maps[index, i], s_colorMeanB, s_colorDeviationB),
                        _ => Normalize(maps[index, i], s_colorMean, s_colorDeviation),
                    };
                }

                for (int y = 0; y < bitmap.Height; y++)
                {
                    for (int x = 0; x < bitmap.Width; x++)
                    {
                        if (maps.GetLength(1) == 1)
                        {
                            bitmap.SetPixel(x, bitmap.Height - y - 1, System.Drawing.Color.FromArgb(Math.Clamp((int)(normalizedMaps[0][y * bitmap.Width + x] * 255), 0, 255), System.Drawing.Color.White));
                        }
                        else
                        {
                            bitmap.SetPixel(x, bitmap.Height - y - 1, System.Drawing.Color.FromArgb(
                                Math.Clamp((int)(normalizedMaps[0][y * bitmap.Width + x] * 255), 0, 255),
                                Math.Clamp((int)(normalizedMaps[1][y * bitmap.Width + x] * 255), 0, 255), 
                                Math.Clamp((int)(normalizedMaps[2][y * bitmap.Width + x] * 255), 0, 255))
                                );
                        }
                    }
                }

                return bitmap;
            }
            return null;
        }

        /// <summary>
        /// Creates a <see cref="Bitmap"/> representation of the <see cref="FeatureMap"/>.
        /// The map is normalized to fit in the range of <see cref="System.Drawing.Color"/>.
        /// </summary>
        /// <param name="setNormalized">If true, the <see cref="FeatureMap"/> will be set to the normalized values.</param>
        /// <returns>Returns the <see cref="FeatureMap"/> as a <see cref="Bitmap"/>.</returns>
        public static Bitmap ConstructNormalizedBitmap(FeatureMap[] maps)
        {
            if (OperatingSystem.IsWindows())
            {
                Bitmap bitmap = new(maps[0].Width, maps[0].Length);

                float[][] normalizedMaps = new float[maps.Length][];

                for (int i = 0; i < maps.Length; i++)
                {
                    normalizedMaps[i] = i switch
                    {
                        0 => Normalize(maps[i], s_colorMeanR, s_colorDeviationR),
                        1 => Normalize(maps[i], s_colorMeanG, s_colorDeviationG),
                        2 => Normalize(maps[i], s_colorMeanB, s_colorDeviationB),
                        _ => Normalize(maps[i], s_colorMean, s_colorDeviation),
                    };
                }

                for (int y = 0; y < bitmap.Height; y++)
                {
                    for (int x = 0; x < bitmap.Width; x++)
                    {
                        if (maps.Length == 1)
                        {
                            bitmap.SetPixel(x, bitmap.Height - y - 1, System.Drawing.Color.FromArgb(Math.Clamp((int)(normalizedMaps[0][y * bitmap.Width + x] * 255), 0, 255), System.Drawing.Color.White));
                        }
                        else if(maps.Length == 3)
                        {
                            bitmap.SetPixel(x, bitmap.Height - y - 1, System.Drawing.Color.FromArgb(
                                Math.Clamp((int)(normalizedMaps[0][y * bitmap.Width + x] * 255), 0, 255),
                                Math.Clamp((int)(normalizedMaps[1][y * bitmap.Width + x] * 255), 0, 255),
                                Math.Clamp((int)(normalizedMaps[2][y * bitmap.Width + x] * 255), 0, 255))
                                );
                        }
                        else
                        {
                            bitmap.SetPixel(x, bitmap.Height - y - 1, System.Drawing.Color.FromArgb(
                                Math.Clamp((int)(normalizedMaps[0][y * bitmap.Width + x] * 255), 0, 255),
                                Math.Clamp((int)(normalizedMaps[1][y * bitmap.Width + x] * 255), 0, 255),
                                Math.Clamp((int)(normalizedMaps[2][y * bitmap.Width + x] * 255), 0, 255),
                                Math.Clamp((int)(normalizedMaps[3][y * bitmap.Width + x] * 255), 0, 255))
                                );
                        }
                    }
                }

                return bitmap;
            }
            return null;
        }

        public static Bitmap ConstructBitmap(FeatureMap[] maps)
        {
            if (OperatingSystem.IsWindows())
            {
                Bitmap bitmap = new(maps[0].Width, maps[0].Length);

                for (int y = 0; y < bitmap.Height; y++)
                {
                    for (int x = 0; x < bitmap.Width; x++)
                    {
                        if (maps.Length == 1)
                        {
                            bitmap.SetPixel(x, bitmap.Height - y - 1, System.Drawing.Color.FromArgb(Math.Clamp((int)(maps[0][x,y] * 255), 0, 255), System.Drawing.Color.White));
                        }
                        else if (maps.Length == 3)
                        {
                            bitmap.SetPixel(x, bitmap.Height - y - 1, System.Drawing.Color.FromArgb(
                                Math.Clamp((int)(maps[0][x, y] * 127.5 + 127.5), 0, 255),
                                Math.Clamp((int)(maps[1][x, y] * 127.5 + 127.5), 0, 255),
                                Math.Clamp((int)(maps[2][x, y] * 127.5 + 127.5), 0, 255))
                                );
                        }
                        else
                        {
                            bitmap.SetPixel(x, bitmap.Height - y - 1, System.Drawing.Color.FromArgb(
                                Math.Clamp((int)(maps[3][x, y] * 127.5 + 127.5), 0, 255),
                                Math.Clamp((int)(maps[0][x, y] * 127.5 + 127.5), 0, 255),
                                Math.Clamp((int)(maps[1][x, y] * 127.5 + 127.5), 0, 255),
                                Math.Clamp((int)(maps[2][x, y] * 127.5 + 127.5), 0, 255))
                                );
                        }
                    }
                }

                return bitmap;
            }
            return null;
        }

        public static Bitmap ConstructBitmap(FeatureMap[] maps, FeatureMap[] source)
        {
            if (OperatingSystem.IsWindows())
            {
                PixelPallet(source, maps);

                Bitmap bitmap = new(maps[0].Width, maps[0].Length);

                for (int y = 0; y < bitmap.Height; y++)
                {
                    for (int x = 0; x < bitmap.Width; x++)
                    {
                        if (maps.Length == 1)
                        {
                            bitmap.SetPixel(x, bitmap.Height - y - 1, System.Drawing.Color.FromArgb(Math.Clamp((int)(maps[0][x, y] * 255), 0, 255), System.Drawing.Color.White));
                        }
                        else if (maps.Length == 3)
                        {
                            bitmap.SetPixel(x, bitmap.Height - y - 1, System.Drawing.Color.FromArgb(
                                Math.Clamp((int)(maps[0][x, y] * 127.5 + 127.5), 0, 255),
                                Math.Clamp((int)(maps[1][x, y] * 127.5 + 127.5), 0, 255),
                                Math.Clamp((int)(maps[2][x, y] * 127.5 + 127.5), 0, 255))
                                );
                        }
                        else
                        {
                            bitmap.SetPixel(x, bitmap.Height - y - 1, System.Drawing.Color.FromArgb(
                                Math.Clamp((int)(maps[3][x, y] * 127.5 + 127.5), 0, 255),
                                Math.Clamp((int)(maps[0][x, y] * 127.5 + 127.5), 0, 255),
                                Math.Clamp((int)(maps[1][x, y] * 127.5 + 127.5), 0, 255),
                                Math.Clamp((int)(maps[2][x, y] * 127.5 + 127.5), 0, 255))
                                );
                        }
                    }
                }

                return bitmap;
            }
            return null;
        }

        /// <summary>
        /// Converts the <see cref="FeatureMap"/> to a <see cref="Bitmap"/> and then saves the image to file as a png.
        /// </summary>
        /// <param name="file">The file to save the image to.</param>
        public void PrintFeatureMap(string file)
        {
            Bitmap image = ConstructBitmap(new FeatureMap[,] { { this } }, 0);

            if (OperatingSystem.IsWindows())
            {
                try
                {
                    image.Save(file, System.Drawing.Imaging.ImageFormat.Png);
                }
                catch (System.Exception e)
                {
                    Console.WriteLine("Error occured when trying to save image: " + file + "\n" + e.ToString());
                }
            }
        }

        /// <summary>
        /// Sums the total <see cref="Color"/> value of the entire <see cref="FeatureMap"/>, used to find the mean <see cref="Color"/>.
        /// (Depending on the size of <see cref="FeatureMap"/>, this can result in overflow errors. Might be preferable to average over
        /// chunks instead of the full map all at once).
        /// </summary>
        /// <returns>Returns the sum of every <see cref="Color"/> in the <see cref="FeatureMap"/>.</returns>
        public float Sum()
        {
            float color = 0;
            for (int i = 0; i < Length; i++)
            {
                for (int j = 0; j < Width; j++)
                {
                    color += _tensor[i * Width + j];
                }
            }

            return color;
        }

        /// <summary>
        /// Normalizes the <see cref="FeatureMap"/>.
        /// </summary>
        /// <returns>Returns the normalized <see cref="FeatureMap"/> as a single dimensional array of <see cref="Color"/>s.</returns>
        public static float[] Normalize(FeatureMap map, float normalMean, float normalDeviation)
        {

            var deviceSum = GPUManager.Accelerator.Allocate1D<float>(1);
            deviceSum.MemSetToZero();

            Index1D index = new(map.Area);

            s_sumAction(index, map.GetArrayView<float>(), deviceSum.View);
            GPUManager.Accelerator.Synchronize();

            map.DecrementLiveCount();
            float[] value = new float[1];
            deviceSum.CopyToCPU(value);

            float mean = value[0] / map.Area;

            var deviceMean = GPUManager.Accelerator.Allocate1D(new float[] { mean });
            var deviceVariance = GPUManager.Accelerator.Allocate1D<float>(1);
            deviceVariance.MemSetToZero();

            s_varianceAction(index, map.GetArrayView<float>(), deviceMean.View, deviceVariance.View);
            GPUManager.Accelerator.Synchronize();

            map.DecrementLiveCount();

            deviceVariance.CopyToCPU(value);

            float sigma = float.Pow(value[0] / map.Area + Utility.ASYMPTOTEERRORCORRECTION, 0.5f);

            var deviceValues = GPUManager.Accelerator.Allocate1D(new float[] { mean, normalDeviation / sigma, normalMean });

            var deviceOutput = GPUManager.Accelerator.Allocate1D<float>(map.Area);


            s_normalizeAction(new Index1D(map.Area), map.GetArrayView<float>(), deviceOutput.View, deviceValues.View);
            GPUManager.Accelerator.Synchronize();

            map.DecrementLiveCount();

            float[] Normalized = new float[map.Area];
            deviceOutput.CopyToCPU(Normalized);

            deviceSum.Dispose();
            deviceOutput.Dispose();
            deviceMean.Dispose();
            deviceVariance.Dispose();
            deviceValues.Dispose();

            map.DeCache();

            return Normalized;
        }

        private static Action<Index1D, ArrayView<float>, ArrayView<float>> s_sumAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(MeanKernel);
        private static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> s_varianceAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VarianceKernel);
        private static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> s_normalizeAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(NormalizeKernel);

        private static void NormalizeKernel(Index1D index, ArrayView<float> input, ArrayView<float> normalized, ArrayView<float> values)
        {
            normalized[index] = (input[index] - values[0]) * values[1] + values[2];
        }

        private static void MeanKernel(Index1D index, ArrayView<float> input, ArrayView<float> mean)
        {
            Atomic.Add(ref mean[0], input[index.X]);
        }

        private static void VarianceKernel(Index1D index, ArrayView<float> input, ArrayView<float> mean, ArrayView<float> variance)
        {
            float difference = input[index.X] - mean[0];
            Atomic.Add(ref variance[0], difference * difference);
        }
    }
}