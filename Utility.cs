using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers;
using ConvolutionalNeuralNetwork.Layers.Weighted;

namespace ConvolutionalNeuralNetwork
{
    /// <summary>
    /// The <see cref="Utility"/> class contains a miscellaenous functions, and consants.
    /// </summary>
    public static class Utility
    {
        //Used to avoid divide by zero or log of zero going to infinity.
        public const float ASYMPTOTEERRORCORRECTION = 1e-7f;

        /// <value>Color with very small values. Used to avoid asymptotic behaviour when a value goes to zero.</value>
        public static Color AsymptoteErrorColor { get; } = new(ASYMPTOTEERRORCORRECTION);

        /// <value>A single <see cref="System.Random"/> for number generation throughout the project. For some functions it is inconvenient
        /// to pass a single <see cref="System.Random"/> but creating multiple in quick succession led to them sometimes being seeded with
        /// the same values, leading to stretches of the same value being generated.</value>
        public static Random Random { get; } = new Random();

        /// <summary>
        /// Tests whether the backpropogation of a <see cref="Layer"/> is accurate to it's expected value. Used to diagnose issues with a <see cref="Layer"/>s
        /// propagation.
        /// </summary>
        /// <param name="layer">The <see cref="Layer"/> to be tested.</param>
        public static void GradientCheck(ILayer layer, int inputDimensions, int outputDimensions, int inputSize, int batchSize)
        {
            FeatureMap[] inputs = new FeatureMap[inputDimensions * batchSize];
            Shape inputShapes = new(inputSize, inputSize, inputDimensions);
            for (int i = 0; i < inputDimensions * batchSize; i++)
            {
                inputs[i] = new(inputSize, inputSize);
                for (int j = 0; j < inputSize; j++)
                {
                    for (int k = 0; k < inputSize; k++)
                    {
                        inputs[i][j, k] = i * 0.5f + (i + 1) * j - k;
                    }
                }
            }

            IOBuffers buffer = new();
            IOBuffers complimentBuffer = new();
            complimentBuffer.OutputDimensionArea(inputDimensions * inputSize * inputSize);

            Shape outputShape = layer.Startup(inputShapes, buffer, batchSize);
            FeatureMap[] outputs = new FeatureMap[outputDimensions * batchSize];
            for(int i = 0; i < outputDimensions * batchSize; i++)
            {
                outputs[i] = new FeatureMap(outputShape);
            }
            buffer.Allocate(batchSize);
            complimentBuffer.Allocate(batchSize);
            IOBuffers.SetCompliment(buffer, complimentBuffer);
            for (int i = 0; i < inputDimensions * batchSize; i++)
            {
                inputs[i].CopyToBuffer(buffer.Input.SubView(inputShapes.Area * i, inputShapes.Area));
            }

            layer.Forward(batchSize);
            if (layer is IUnchangedLayer)
            {
                for (int i = 0; i < outputDimensions * batchSize; i++)
                {
                    outputs[i].SyncCPU(buffer.Input.SubView(outputShape.Area * i, outputShape.Area));
                    new FeatureMap(outputs[i].Width, outputs[i].Length, 1).CopyToBuffer(buffer.Gradient.SubView(outputShape.Area * i, outputShape.Area));
                }
            }
            else
            {
                for (int i = 0; i < outputDimensions * batchSize; i++)
                {
                    outputs[i].SyncCPU(buffer.Output.SubView(outputShape.Area * i, outputShape.Area));
                    new FeatureMap(outputs[i].Width, outputs[i].Length, 1).CopyToBuffer(buffer.InGradient.SubView(outputShape.Area * i, outputShape.Area));
                }
            }
            layer.Backwards(batchSize, true);
            FeatureMap[] outGradients = new FeatureMap[inputDimensions * batchSize];
            for (int i = 0; i < inputDimensions * batchSize; i++)
            {
                outGradients[i] = new FeatureMap(inputSize, inputSize);
                outGradients[i].SyncCPU(buffer.OutGradient.SubView(inputShapes.Area * i, inputShapes.Area));
            }

            FeatureMap[] testOutput = new FeatureMap[outputDimensions * batchSize];
            for (int i = 0; i < outputDimensions * batchSize; i++)
            {

                testOutput[i] = new(outputs[i].Width, outputs[i].Length);
            }

            float h = 0.001f;

            

            for (int i = 0; i < inputDimensions * batchSize; i++)
            {
                for (int j = 0; j < inputDimensions * batchSize; j++)
                {
                    inputs[j].CopyToBuffer(buffer.Input.SubView(inputShapes.Area * j, inputShapes.Area));
                }
                for (int k = 0; k < inputSize; k++)
                {
                    for (int j = 0; j < inputSize; j++)
                    {

                        inputs[i][j, k] += h;
                        for (int i2 = 0; i2 < inputDimensions * batchSize; i2++)
                        {
                            inputs[i2].CopyToBuffer(buffer.Input.SubView(inputShapes.Area * i2, inputShapes.Area));
                        }


                        layer.Forward(batchSize);

                        float testGradient = 0;
                        for (int i2 = 0; i2 < outputDimensions * batchSize; i2++)
                        {
                            if (layer is IUnchangedLayer)
                            {
                                testOutput[i2].SyncCPU(buffer.Input.SubView(outputShape.Area * i2, outputShape.Area));
                            }
                            else
                            {
                                testOutput[i2].SyncCPU(buffer.Output.SubView(outputShape.Area * i2, outputShape.Area));
                            }
                            for (int k2 = 0; k2 < testOutput[i2].Length; k2++)
                            {
                                for (int j2 = 0; j2 < testOutput[i2].Width; j2++)
                                {


                                    testGradient += (testOutput[i2][j2, k2] - outputs[i2][j2, k2]) / h;

                                }
                            }
                        }
                        if (MathF.Abs(outGradients[i][j, k] - testGradient) > Math.Max(0.01, testGradient * 0.001))
                        {
                            Console.WriteLine($"Expected Gradient: {outGradients[i][j, k]:f4} \t Test Gradient: {testGradient:f4}");
                            Console.ReadLine();
                        }
                        inputs[i][j, k] = i * 0.5f + (i + 1) * j - k;
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