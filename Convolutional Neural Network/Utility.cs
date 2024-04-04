namespace ConvolutionalNeuralNetwork
{
    /// <summary>
    /// The <see cref="Utility"/> class contains a miscellaneous functions, and constants.
    /// </summary>
    public static class Utility
    {

        /// <value>Correction term used to avoid divide by zero or log of zero going to infinity.</value>
        public const float ASYMPTOTE_ERROR_CORRECTION = 1e-7f;

        /// <value>A single <see cref="System.Random"/> for number generation throughout the project. For some functions it is inconvenient
        /// to pass a single <see cref="System.Random"/> but creating multiple in quick succession led to them sometimes being seeded with
        /// the same values, leading to stretches of the same value being generated.</value>
        public static Random Random { get; } = new();

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
    }
}