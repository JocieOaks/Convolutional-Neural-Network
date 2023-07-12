using ILGPU;
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

        /// <value>A Cuda <see cref="ILGPU.Runtime.Accelerator"/> for running <see cref="ILGPU"/> kernals.</value>
        public static Accelerator Accelerator { get; } = Context.CreateCudaAccelerator(0);

        /// <value>Color with very small values. Used to avoid asymptotic behaviour when a value goes to zero.</value>
        public static DataTypes.Color AsymptoteErrorColor { get; } = new(ASYMPTOTEERRORCORRECTION);

        /// <value>A Cuda <see cref="ILGPU.Context"/> for running <see cref="ILGPU"/> kernals.</value>
        public static Context Context { get; } = Context.Create(builder => builder.Cuda());

        /// <value>An action for running the cuda kernal <see cref="CopyKernal(Index1D, ArrayView{Color}, ArrayView{Color})"/>.</value>
        public static Action<Index1D, ArrayView<Color>, ArrayView<Color>> CopyAction { get; } = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Color>, ArrayView<Color>>(CopyKernal);

        /// <value>A single <see cref="System.Random"/> for number generation throughout the project. For some functions it is inconvenient
        /// to pass a single <see cref="System.Random"/> but creating multiple in quick succession led to them sometimes being seeded with
        /// the same values, leading to stretches of the same value being generated.</value>
        public static Random Random { get; } = new Random();

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

        /// <summary>
        /// Tests whether the backpropogation of a <see cref="Layer"/> is accurate to it's expected value. Used to diagnose issues with a <see cref="Layer"/>s
        /// propagation.
        /// </summary>
        /// <param name="layer">The <see cref="Layer"/> to be tested.</param>
        public static void GradientCheck(ILayer layer, int dimensionMultiplier)
        {
            int outputDimensions, inputDimensions;
            if (dimensionMultiplier >= 1)
            {
                inputDimensions = 1;
                outputDimensions = dimensionMultiplier;
            }
            else
            {
                inputDimensions = -dimensionMultiplier;
                outputDimensions = 1;
            }

            FeatureMap[,] inputs = new FeatureMap[inputDimensions, 1];
            for (int i = 0; i < inputDimensions; i++)
            {
                inputs[i, 0] = new(3, 3);
                for (int j = 0; j < 3; j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        inputs[i, 0][j, k] = (i + 1) * new Color(j, k, j - k);
                    }
                }
            }

            IOBuffers buffer = new();
            IOBuffers complimentBuffer = new();
            complimentBuffer.OutputDimensionArea(inputDimensions - 1, 9);

            FeatureMap[,] outputs = layer.Startup(inputs, buffer);
            buffer.Allocate(1);
            complimentBuffer.Allocate(1);
            IOBuffers.SetCompliment(buffer, complimentBuffer);
            for (int i = 0; i < inputDimensions; i++)
            {
                inputs[i, 0].CopyToBuffer(buffer.InputsColor[i, 0]);
            }

            layer.Forward();
            for (int i = 0; i < outputDimensions; i++)
            {
                outputs[i, 0].CopyFromBuffer(buffer.OutputsColor[i, 0]);
                new FeatureMap(outputs[i, 0].Width, outputs[i, 0].Length, Color.One).CopyToBuffer(buffer.InGradientsColor[i, 0]);
            }

            layer.Backwards(1, 1, 1);
            FeatureMap[] outGradients = new FeatureMap[inputDimensions];
            for (int i = 0; i < inputDimensions; i++)
            {
                outGradients[i] = new FeatureMap(3, 3);
                outGradients[i].CopyFromBuffer(buffer.OutGradientsColor[i, 0]);
            }

            FeatureMap[] testOutput = new FeatureMap[outputDimensions];
            for (int i = 0; i < outputDimensions; i++)
            {

                testOutput[i] = new(outputs[i, 0].Width, outputs[i, 0].Length);
            }

            float h = 0.001f;

            for (int i = 0; i < inputDimensions; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        for (int l = 0; l < 3; l++)
                        {
                            Color hColor = l switch
                            {
                                0 => new Color(h, 0, 0),
                                1 => new Color(0, h, 0),
                                2 => new Color(0, 0, h)
                            };

                            inputs[i, 0][j, k] += hColor;
                            for (int i2 = 0; i2 < inputDimensions; i2++)
                            {
                                inputs[i2, 0].CopyToBuffer(buffer.InputsColor[i2, 0]);
                            }

                            layer.Forward();



                            float testGradient = 0;
                            for (int i2 = 0; i2 < outputDimensions; i2++)
                            {
                                testOutput[i2].CopyFromBuffer(buffer.OutputsColor[i2, 0]);
                                for (int j2 = 0; j2 < testOutput[i2].Width; j2++)
                                {
                                    for (int k2 = 0; k2 < testOutput[i2].Length; k2++)
                                    {
                                        for (int l2 = 0; l2 < 3; l2++)
                                        {
                                            testGradient += (testOutput[i2][j2, k2][l2] - outputs[i2, 0][j2, k2][l2]) / h;
                                        }
                                    }
                                }
                            }
                            Console.WriteLine($"Expected Gradient: {outGradients[i][j, k][l]:f4} \t Test Gradient: {testGradient:f4}");
                            inputs[i, 0][j, k] -= hColor;
                        }
                    }
                }
            }
        }

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
    }
}