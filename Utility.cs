﻿using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers;

namespace ConvolutionalNeuralNetwork
{
    /// <summary>
    /// The <see cref="Utility"/> class contains a miscellaenous functions, and consants.
    /// </summary>
    public static class Utility
    {
        //Used to avoid divide by zero or log of zero going to infinity.
        public const float ASYMPTOTEERRORCORRECTION = 1e-6f;

        /// <value>Color with very small values. Used to avoid asymptotic behaviour when a value goes to zero.</value>
        public static DataTypes.Color AsymptoteErrorColor { get; } = new(ASYMPTOTEERRORCORRECTION);

        /// <value>A Cuda <see cref="ILGPU.Context"/> for running <see cref="ILGPU"/> kernals.</value>
        public static Context Context { get; } = Context.Create(builder => builder.Cuda());

        /// <value>A Cuda <see cref="ILGPU.Runtime.Accelerator"/> for running <see cref="ILGPU"/> kernals.</value>
        public static Accelerator Accelerator { get; } = Context.CreateCudaAccelerator(0);

        /// <value>A single <see cref="System.Random"/> for number generation throughout the project. For some functions it is inconvenient
        /// to pass a single <see cref="System.Random"/> but creating multiple in quick succession led to them sometimes being seeded with
        /// the same values, leading to stretches of the same value being generated.</value>
        public static Random Random { get; } = new Random();

        /// <summary>
        /// Generates a random value from a gaussian distribution with the given parameters.
        /// </summary>
        /// <param name="mean">The mean of the distribution.</param>
        /// <param name="stdDev">The standard deviation of the distribution.</param>
        /// <returns>Returns a random number where the total set of generated values will form a gaussian distribution.</returns>
        public static float RandomGauss(float mean, float stdDev)
        {
            float u1 = 1 - (float)Random.NextDouble();
            float u2 = 1 - (float)Random.NextDouble();
            float randStdNormal = MathF.Sqrt(-2 * MathF.Log(u1)) * MathF.Sin(2 * MathF.PI * u2);
            return mean + stdDev * randStdNormal;
        }

        /// <summary>
        /// Performs an <see cref="Action"/> while measuring the length of time that the action takes to complete.
        /// </summary>
        /// <param name="func">The <see cref="Action"/> to be performed and measured.</param>
        /// <param name="processName">The name of the action, for logging purposes.</param>
        /// <param name="print">When true, the time taken for the <see cref="Action"/> to be completed will be printed to the console.</param>
        public static float StopWatch(Action func, string processName, bool print)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            func();
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            if (print)
                Console.WriteLine($"Time: {elapsedMs / 1000f:F3} s {processName}");
            return elapsedMs / 1000f;
        }

        /// <summary>
        /// Transposes a 2D array.
        /// </summary>
        /// <typeparam name="T">The type of objects stored in the array.</typeparam>
        /// <param name="array">The 2D array being transposed.</param>
        /// <returns>Returns a transposed copy of the array, so that the first and second indeces are reversed.</returns>
        /// Note: transposing arrays was necessary for an old implementation of <see cref="Layers.Vectorization"/>; however, this is likely no longer
        /// necessary so this functionality could be removed.
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

        /// <summary>
        /// Kernal for copying values from one <see cref="ArrayView{T}"/> to another.
        /// </summary>
        /// <param name="index">The index to iterate over every element in the two <see cref="ArrayView{T}"/>s.</param>
        /// <param name="input">The <see cref="ArrayView{T}"/> being copied from.</param>
        /// <param name="output">The <see cref="ArrayView{T}"/> being copied to.</param>
        public static void CopyKernal(Index1D index, ArrayView<Color> input, ArrayView<Color> output)
        {
            output[index] = input[index];
        }

        public static Action<Index1D, ArrayView<Color>, ArrayView<Color>> CopyAction { get; } = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Color>, ArrayView<Color>>(CopyKernal);

        /// <summary>
        /// Tests whether the backpropogation of a <see cref="Layer"/> is accurate to it's expected value. Used to diagnose issues with a <see cref="Layer"/>s
        /// propagation.
        /// </summary>
        /// <param name="layer">The <see cref="Layer"/> to be tested.</param>
        public static void GradientTest(ILayer layer)
        {
            FeatureMap input = new(3, 3);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    input[i, j] = new Color(i, j, i - j);
                }
            }

            IOBuffers buffer = new();
            IOBuffers complimentBuffer = new();
            complimentBuffer.OutputDimensionArea(0, 9);

            FeatureMap output = layer.Startup(new FeatureMap[,] { { input } }, buffer)[0, 0];
            buffer.Allocate(1);
            complimentBuffer.Allocate(1);
            IOBuffers.SetCompliment(buffer, complimentBuffer);
            input.CopyToBuffer(buffer.InputsColor[0, 0]);

            layer.Forward();
            output.CopyFromBuffer(buffer.OutputsColor[0, 0]);
            FeatureMap gradient = new(output.Width, output.Length, Color.One);
            gradient[0, 0] = Color.One;
            gradient.CopyToBuffer(buffer.InGradientsColor[0, 0]);
            layer.Backwards(0, 0, 0);
            FeatureMap outGradient = new(3, 3);
            outGradient.CopyFromBuffer(buffer.OutGradientsColor[0, 0]);

            FeatureMap testOutput = new(output.Width, output.Length);

            float h = 0.001f;

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        Color hColor = k switch
                        {
                            0 => new Color(h, 0, 0),
                            1 => new Color(0, h, 0),
                            2 => new Color(0, 0, h)
                        };
                        input[i, j] += hColor;
                        input.CopyToBuffer(buffer.InputsColor[0, 0]);

                        layer.Forward();
                        testOutput.CopyFromBuffer(buffer.OutputsColor[0, 0]);

                        float testGradient = 0;
                        for (int i2 = 0; i2 < output.Width; i2++)
                        {
                            for (int j2 = 0; j2 < output.Length; j2++)
                            {
                                for (int k2 = 0; k2 < 3; k2++)
                                {
                                    testGradient += (testOutput[i2, j2][k2] - output[i2, j2][k2]) / h;
                                }
                            }
                        }

                        Console.WriteLine($"Expected Gradient: {outGradient[i, j][k]:f4} \t Test Gradient: {testGradient:f4}");
                        input[i, j] -= hColor;
                    }
                }
            }
        }
    }
}