using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Drawing;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    [Serializable]
    public class FeatureMap
    {
        private static Color s_normalMean = new(0.5f);
        private static Color s_normalStandardDeviation = new(0.23f);
        [JsonProperty] private Color[] _map;

        public FeatureMap(int width, int length)
        {
            Width = width;
            Length = length;

            _map = new Color[width * length];
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

        [JsonConstructor]
        private FeatureMap()
        {
        }

        [JsonIgnore] public int Area => _map.Length;

        [JsonIgnore] public int FloatLength => _map.Length * 3;

        [JsonProperty] public int Length { get; private set; }

        [JsonProperty] public int Width { get; private set; }

        public Color this[int x, int y]
        {
            get => _map[y * Width + x];
            set => _map[y * Width + x] = value;
        }

        public static FeatureMap Random(int width, int length)
        {
            FeatureMap map = new(width, length);
            for (int y = 0; y < length; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    map[x, y] = Color.RandomGauss(0.5f, 0.2f).Clamp(1);
                }
            }
            return map;
        }

        public bool HasNaN()
        {
            for (int i = 0; i < Area; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    if (float.IsNaN(_map[i][j]))
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        public MemoryBuffer1D<Color, Stride1D.Dense> Allocate(Accelerator accelerator)
        {
            return accelerator.Allocate1D(_map);
        }

        public MemoryBuffer1D<Color, Stride1D.Dense> AllocateEmpty(Accelerator accelerator)
        {
            return accelerator.Allocate1D<Color>(Area);
        }

        public MemoryBuffer1D<float, Stride1D.Dense> AllocateFloat(Accelerator accelerator)
        {
            return accelerator.Allocate1D<float>(FloatLength);
        }

        public Color Average()
        {
            return Sum() / Area;
        }

        public float AverageMagnitude()
        {
            return SumMagnitude() / Area;
        }

        public Bitmap ConstructBitmap(Accelerator accelerator, bool setNormalized = false)
        {
            Bitmap bitmap = new(Width, Length);

            Color[] normalizedMap = Normalize(accelerator);

            if (setNormalized)
                _map = normalizedMap;

            for (int y = 0; y < Length; y++)
            {
                for (int x = 0; x < Width; x++)
                {
                    bitmap.SetPixel(x, Length - y - 1, (System.Drawing.Color)(normalizedMap[y * Width + x] * 255));
                }
            }

            return bitmap;
        }

        public static FeatureMap FromBitmap(Bitmap bitmap)
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

        public void CopyFromBuffer(MemoryBuffer1D<Color, Stride1D.Dense> buffer)
        {
            buffer.CopyToCPU(_map);
        }

        public void CopyFromBuffer(MemoryBuffer1D<float, Stride1D.Dense> buffer)
        {
            unsafe
            {
                fixed (void* ptr = &_map[0])
                {
                    Span<float> span = new(ptr, Area * 3);
                    buffer.AsArrayView<float>(0, Area * 3).CopyToCPU(span);
                }
            }
        }

        public Color Sum()
        {
            Color color = new();
            for (int i = 0; i < Length; i++)
            {
                for (int j = 0; j < Width; j++)
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

        public void PrintFeatureMap(string file, Accelerator accelerator)
        {
            Bitmap image = ConstructBitmap(accelerator);
            try
            {
                image.Save(file, System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (System.Exception e)
            {
                Console.WriteLine("Error occured when trying to save image: " + file + "\n" + e.ToString());
            }
        }

        private Color[] Normalize(Accelerator accelerator)
        {
            var sumKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<float>>(MeanKernal);
            var varianceKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>>(VarianceKernal);
            var normalizeKernal = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>>(NormalizeKernal);

            var deviceSum = accelerator.Allocate1D<float>(3);

            Index2D index = new(Area, 3);

            var deviceInput = Allocate(accelerator);

            //Console.Write("Sum");
            sumKernal(index, deviceInput.View, deviceSum.View);

            accelerator.Synchronize();

            Color mean = (Color)deviceSum / Area;

            var deviceMean = accelerator.Allocate1D(new Color[] { mean });
            var deviceVariance = accelerator.Allocate1D<float>(3);

            //Console.WriteLine("Variance");
            varianceKernal(index, deviceInput.View, deviceMean.View, deviceVariance.View);

            accelerator.Synchronize();

            Color sigma = Color.Pow((Color)deviceVariance / Area + new Color(ConvolutionalNeuralNetwork.Utility.ASYMPTOTEERRORFACTOR), 0.5f);

            var deviceValues = accelerator.Allocate1D(new Color[] { mean, s_normalStandardDeviation / sigma, s_normalMean });

            var deviceOutput = AllocateEmpty(accelerator);
            //Console.WriteLine("Normalize");
            normalizeKernal(new Index1D(Area), deviceInput.View, deviceOutput.View, deviceValues.View);

            accelerator.Synchronize();

            Color[] Normalized = new Color[Area];
            deviceOutput.CopyToCPU(Normalized);

            deviceSum.Dispose();
            deviceInput.Dispose();
            deviceOutput.Dispose();
            deviceMean.Dispose();
            deviceVariance.Dispose();
            deviceValues.Dispose();

            return Normalized;

            void NormalizeKernal(Index1D index, ArrayView<Color> input, ArrayView<Color> normalized, ArrayView<Color> values)
            {
                normalized[index] = (input[index] - values[0]) * values[1] + values[2];
            }

            void MeanKernal(Index2D index, ArrayView<Color> input, ArrayView<float> mean)
            {
                Atomic.Add(ref mean[index.Y], input[index.X][index.Y]);
            }

            void VarianceKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> mean, ArrayView<float> variance)
            {
                float difference = input[index.X][index.Y] - mean[0][index.Y];
                Atomic.Add(ref variance[index.Y], difference * difference);
            }
        }
    }
}