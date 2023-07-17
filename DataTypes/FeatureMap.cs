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
        private readonly static float s_colorMean = -0.7386f;
        private readonly static float s_colorDeviation = 0.6026f;

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
                            map[x, bitmap.Height - y - 1] = bitmap.GetPixel(x, y).GetBrightness();
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
        public Bitmap ConstructBitmap(bool setNormalized = false)
        {
            if (OperatingSystem.IsWindows())
            {
                Bitmap bitmap = new(Width, Length);

                float[] normalizedMap = Normalize(s_colorMean, s_colorDeviation);

                if (setNormalized)
                    _tensor = normalizedMap;

                for (int y = 0; y < Length; y++)
                {
                    for (int x = 0; x < Width; x++)
                    {
                        bitmap.SetPixel(x, Length - y - 1, System.Drawing.Color.FromArgb(Math.Clamp((int)(normalizedMap[y * Width + x] * 255), 0, 255), System.Drawing.Color.White));
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
        public float[] Normalize(float normalMean, float normalDeviation)
        {

            var deviceSum = GPUManager.Accelerator.Allocate1D<float>(1);
            deviceSum.MemSetToZero();

            Index1D index = new(Area);

            s_sumAction(index, GetArrayView<float>(), deviceSum.View);
            GPUManager.Accelerator.Synchronize();

            DecrementLiveCount();
            float[] value = new float[1];
            deviceSum.CopyToCPU(value);

            float mean = value[0] / Area;

            var deviceMean = GPUManager.Accelerator.Allocate1D(new float[] { mean });
            var deviceVariance = GPUManager.Accelerator.Allocate1D<float>(1);
            deviceVariance.MemSetToZero();

            s_varianceAction(index, GetArrayView<float>(), deviceMean.View, deviceVariance.View);
            GPUManager.Accelerator.Synchronize();

            DecrementLiveCount();

            deviceVariance.CopyToCPU(value);

            float sigma = float.Pow(value[0] / Area + Utility.ASYMPTOTEERRORCORRECTION, 0.5f);

            var deviceValues = GPUManager.Accelerator.Allocate1D(new float[] { mean, normalDeviation / sigma, normalMean });

            var deviceOutput = GPUManager.Accelerator.Allocate1D<float>(Area);


            s_normalizeAction(new Index1D(Area), GetArrayView<float>(), deviceOutput.View, deviceValues.View);
            GPUManager.Accelerator.Synchronize();

            DecrementLiveCount();

            float[] Normalized = new float[Area];
            deviceOutput.CopyToCPU(Normalized);

            deviceSum.Dispose();
            deviceOutput.Dispose();
            deviceMean.Dispose();
            deviceVariance.Dispose();
            deviceValues.Dispose();

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