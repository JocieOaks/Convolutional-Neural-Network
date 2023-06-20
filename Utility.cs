using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace ConvolutionalNeuralNetwork
{
    /// <summary>
    /// The <see cref="Utility"/> class contains a variety of useful functions, and consants.
    /// </summary>
    public static class Utility
    {
        //Used to avoid divide by zero or log of zero going to infinity.
        public const float ASYMPTOTEERRORCORRECTION = 1e-6f;

        /// <value>Color with very small values. Used to avoid asymptotic behaviour when a value goes to zero.</value>
        public static DataTypes.Color AsymptoteErrorColor { get; } = new(ASYMPTOTEERRORCORRECTION);

        public static Context Context { get; } = Context.Create(builder => builder.Cuda());
        public static Accelerator Accelerator { get; } = Context.CreateCudaAccelerator(0);
        public static Random Random { get; } = new Random();

        public static float RandomGauss(float mean, float stdDev)
        {
            float u1 = 1 - (float)Random.NextDouble();
            float u2 = 1 - (float)Random.NextDouble();
            float randStdNormal = MathF.Sqrt(-2 * MathF.Log(u1)) * MathF.Sin(2 * MathF.PI * u2);
            return mean + stdDev * randStdNormal;
        }

        public static void StopWatch(Action func, string processName, bool print)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            func();
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            if (print)
                Console.WriteLine($"Time: {elapsedMs / 1000f:F3} s {processName}");
        }

        public static T[][] TransposeArray<T>(T[][] array)
        {
            T[][] transposed = new T[array[0].Length][];
            for (int i = 0; i < transposed.Length; i++)
            {
                transposed[i] = new T[array.Length];
                for (int j = 0; j < transposed[i].Length; j++)
                {
                    transposed[i][j] = array[j][i];
                }
            }
            return transposed;
        }

        public static T[,] TransposeArray<T>(T[,] array)
        {
            T[,] transposed = new T[array.GetLength(1), array.GetLength(0)];
            for (int i = 0; i < transposed.GetLength(0); i++)
            {
                for (int j = 0; j < transposed.GetLength(1); j++)
                {
                    transposed[i, j] = array[j, i];
                }
            }
            return transposed;
        }
    }
}