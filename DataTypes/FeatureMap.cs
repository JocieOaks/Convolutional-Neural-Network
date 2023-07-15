using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Drawing;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="FeatureMap"/> class contains color data for the input and ouputs of layers and their gradients.
    /// (In some cases <see cref="FeatureMap"/> is used as an n x m x 3 tensor, because that is what it fundamentally is. However, this
    /// can be confusing, so it may be better to create a Tensor class of which <see cref="FeatureMap"/> is a child.)
    /// </summary>
    [Serializable]
    public class FeatureMap : ColorTensor
    {
        private readonly static Color s_colorMean = new(0.5f);
        private readonly static Color s_colorDeviation = new(MathF.Sqrt(1f / 12));

        /// <summary>
        /// Initializes a new <see cref="FeatureMap"/> with the given dimensions.
        /// </summary>
        /// <param name="width">The width of the <see cref="FeatureMap"/>.</param>
        /// <param name="length">The length of the <see cref="FeatureMap"/>.</param>
        public FeatureMap(int width, int length) : base(width, length) { }

        public FeatureMap(int width, int length, Color color) : base(width, length)
        {
            for(int i = 0; i < Area; i++)
            {
                _tensor[i] = color;
            }
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor]
        private FeatureMap()
        {
        }

        /// <summary>
        /// Creates a new <see cref="FeatureMap"/> from a given <see cref="Bitmap"/>.
        /// </summary>
        /// <param name="bitmap">The <see cref="Bitmap"/> being converted.</param>
        /// <returns>Returns the <see cref="FeatureMap"/> representation of the <see cref="Bitmap"/>.</returns>
        public static FeatureMap FromBitmap(Bitmap bitmap)
        {
            if (OperatingSystem.IsWindows())
            {
                FeatureMap map = new(bitmap.Width, bitmap.Height);
                {
                    for (int y = 0; y < bitmap.Height; y++)
                    {
                        for (int x = 0; x < bitmap.Width; x++)
                        {
                            map[x, bitmap.Height - y - 1] = (Color)bitmap.GetPixel(x, y);
                        }
                    }
                }
                return map;
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
                    map[x, y] = Color.RandomGauss(0.5f, 0.2f).Clip(1);
                }
            }
            return map;
        }

        /// <summary>
        /// Calculates the average <see cref="Color"/> of the <see cref="FeatureMap"/>.
        /// </summary>
        /// <returns>Returns the <see cref="Color"/>.</returns>
        public Color Average()
        {
            return Sum() / Area;
        }

        /// <summary>
        /// Creates a <see cref="Bitmap"/> representation of the <see cref="FeatureMap"/>.
        /// The map is normalized to fit in the range of <see cref="System.Drawing.Color"/>.
        /// </summary>
        /// <param name="setNormalized">If true, the <see cref="FeatureMap"/> will be set to the normalized values.</param>
        /// <returns>Returns the <see cref="FeatureMap"/> as a <see cref="Bitmap"/>.</returns>
        public Bitmap ConstructBitmap(bool setNormalized = false)
        {
            if (OperatingSystem.IsWindows())
            {
                Bitmap bitmap = new(Width, Length);

                Color[] normalizedMap = Normalize(new(-.7386f), new(0.6026f));

                if (setNormalized)
                    _tensor = normalizedMap;

                for (int y = 0; y < Length; y++)
                {
                    for (int x = 0; x < Width; x++)
                    {
                        bitmap.SetPixel(x, Length - y - 1, (System.Drawing.Color)(normalizedMap[y * Width + x] * 255));
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
            Bitmap image = ConstructBitmap();

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
        public Color Sum()
        {
            Color color = new();
            for (int i = 0; i < Length; i++)
            {
                for (int j = 0; j < Width; j++)
                {
                    color += _tensor[i * Width + j];
                }
            }

            return color;
        }

        public (Color, Color) MeanVariance()
        {

            var deviceSum = GPUManager.Accelerator.Allocate1D<float>(3);
            deviceSum.MemSetToZero();

            Index2D index = new(Area, 3);

            s_sumAction(index, GetArrayView<Color>(), deviceSum.View);
            GPUManager.Accelerator.Synchronize();

            Color mean = (Color)deviceSum / Area;

            var deviceMean = GPUManager.Accelerator.Allocate1D(new Color[] { mean });
            var deviceVariance = GPUManager.Accelerator.Allocate1D<float>(3);
            deviceVariance.MemSetToZero();

            s_varianceAction(index, GetArrayView<Color>(), deviceMean.View, deviceVariance.View);
            GPUManager.Accelerator.Synchronize();

            Color sigma = Color.Pow((Color)deviceVariance / Area + Utility.AsymptoteErrorColor, 0.5f);

            deviceSum.Dispose();
            deviceMean.Dispose();
            deviceVariance.Dispose();

            return (mean, sigma);
        }

        /// <summary>
        /// Normalizes the <see cref="FeatureMap"/>.
        /// </summary>
        /// <returns>Returns the normalized <see cref="FeatureMap"/> as a single dimensional array of <see cref="Color"/>s.</returns>
        public Color[] Normalize(Color normalMean, Color normalDeviation)
        {

            var deviceSum = GPUManager.Accelerator.Allocate1D<float>(3);
            deviceSum.MemSetToZero();

            Index2D index = new(Area, 3);

            s_sumAction(index, GetArrayView<Color>(), deviceSum.View);
            GPUManager.Accelerator.Synchronize();

            Color mean = (Color)deviceSum / Area;

            var deviceMean = GPUManager.Accelerator.Allocate1D(new Color[] { mean });
            var deviceVariance = GPUManager.Accelerator.Allocate1D<float>(3);
            deviceVariance.MemSetToZero();

            s_varianceAction(index, GetArrayView<Color>(), deviceMean.View, deviceVariance.View);
            GPUManager.Accelerator.Synchronize();

            Color sigma = Color.Pow((Color)deviceVariance / Area + Utility.AsymptoteErrorColor, 0.5f);

            var deviceValues = GPUManager.Accelerator.Allocate1D(new Color[] { mean, normalDeviation / sigma, normalMean });

            var deviceOutput = GPUManager.Accelerator.Allocate1D<Color>(Area);


            s_normalizeAction(new Index1D(Area), GetArrayView<Color>(), deviceOutput.View, deviceValues.View);
            GPUManager.Accelerator.Synchronize();

            Color[] Normalized = new Color[Area];
            deviceOutput.CopyToCPU(Normalized);

            deviceSum.Dispose();
            deviceOutput.Dispose();
            deviceMean.Dispose();
            deviceVariance.Dispose();
            deviceValues.Dispose();

            return Normalized;


        }

        private static Action<Index2D, ArrayView<Color>, ArrayView<float>> s_sumAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<float>>(MeanKernel);
        private static Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>> s_varianceAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>>(VarianceKernel);
        private static Action<Index1D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>> s_normalizeAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>>(NormalizeKernel);

        private static void NormalizeKernel(Index1D index, ArrayView<Color> input, ArrayView<Color> normalized, ArrayView<Color> values)
        {
            normalized[index] = (input[index] - values[0]) * values[1] + values[2];
        }

        private static void MeanKernel(Index2D index, ArrayView<Color> input, ArrayView<float> mean)
        {
            Atomic.Add(ref mean[index.Y], input[index.X][index.Y]);
        }

        private static void VarianceKernel(Index2D index, ArrayView<Color> input, ArrayView<Color> mean, ArrayView<float> variance)
        {
            float difference = input[index.X][index.Y] - mean[0][index.Y];
            Atomic.Add(ref variance[index.Y], difference * difference);
        }
    }
}