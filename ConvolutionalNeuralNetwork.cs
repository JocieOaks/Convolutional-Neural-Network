using ILGPU;
using ILGPU.IR.Analyses.ControlFlowDirection;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;
using System.Drawing;
using System.IO;
using System.Reflection.Metadata.Ecma335;

[Serializable]
public abstract partial class ConvolutionalNeuralNetwork
{
    //Used to avoid divide by zero or log of zero going to infinity.
    public const float ASYMPTOTEERRORFACTOR = 1e-6f; //Used to avoid divide by zero or log of zero going to infinity.

    private const bool PRINTSTOPWATCH = false;

    public static Context Context { get; } = Context.Create(builder => builder.Cuda());
    public static Accelerator Accelerator { get; } = Context.CreateCudaAccelerator(0);

    [JsonConstructor]
    public ConvolutionalNeuralNetwork()
    {
    }

    public static Random Random { get; } = new Random();

    protected int Depth => _layers.Count;

    public static float Accuracy(float[,] matrix)
    {
        int correct = 0;
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            int bestImageIndex = 0;
            float bestImageValue = matrix[i, 0];
            int bestDescriptionIndex = 0;
            float bestDescriptionValue = matrix[0, i];

            for (int j = 0; j < matrix.GetLength(1); j++)
            {
                if (matrix[i, j] > bestImageValue)
                {
                    bestImageIndex = j;
                    bestImageValue = matrix[i, j];
                }
                if (matrix[j, i] > bestDescriptionValue)
                {
                    bestDescriptionIndex = j;
                    bestDescriptionValue = matrix[j, i];
                }
            }
            if (bestImageIndex == i)
                correct++;
            if (bestDescriptionIndex == i)
                correct++;
        }
        return correct / (2f * matrix.GetLength(0));
    }

    public static float Loss(float[,] matrix)
    {
        float loss = 0.0f;
        int length = matrix.GetLength(0);
        for (int i = 0; i < length; i++)
        {
            float totalI = 0;
            float totalD = 0;
            for (int j = 0; j < length; j++)
            {
                totalI += MathF.Exp(2 * matrix[i, j] - 2);
                totalD += MathF.Exp(2 * matrix[j, i] - 2);
            }

            for (int j = 0; j < length; j++)
            {
                if (i == j)
                    loss += MathF.Log(MathF.Exp(2 * matrix[i, j] - 2) * MathF.Exp(2 * matrix[j, i] - 2) / totalD / totalI);
                else
                    loss += MathF.Log((totalD - MathF.Exp(2 * matrix[j, i] - 2)) / totalD) + MathF.Log((totalI - MathF.Exp(2 * matrix[i, j] - 2)) / totalI);
            }
        }
        return -loss / (length * length);
    }

    public static float RandomGauss(float mean, float stdDev)
    {
        float u1 = 1 - (float)Random.NextDouble(); //uniform(0,1] random doubles
        float u2 = 1 - (float)Random.NextDouble();
        float randStdNormal = MathF.Sqrt(-2 * MathF.Log(u1)) * MathF.Sin(2 * MathF.PI * u2); //random normal(0,1)
        return mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)
    }

    public void PrintFeatureMaps(string directory, string name, int batchIndex)
    {
        directory = Path.Combine(directory, name);
        try
        {
            // create the directory the file will be written to if it doesn't already exist
            Directory.CreateDirectory(directory);
        }
        catch (System.Exception e)
        {
            Console.WriteLine("Error occured when trying to create director: " + directory + "\n" + e.ToString());
        }
        Context context = ConvolutionalNeuralNetwork.Context;
        Accelerator accelerator = ConvolutionalNeuralNetwork.Accelerator;
        string layerDirectory;
        for (int i = 0; i < Depth; i++)
        {
            if (_layers[i] is BatchNormalizationLayer)
            {
                layerDirectory = Path.Combine(directory, $"{i} {_layers[i].Name}");
                Directory.CreateDirectory(layerDirectory);
                for (int j = 0; j < _layers[i].OutputDimensions; j++)
                {
                    PrintFeatureMap(_layers[i].Outputs[j, batchIndex], Path.Combine(layerDirectory, $"{name} {j}.png"), accelerator);
                }
            }
        }
    }

    private void PrintFeatureMap(FeatureMap map, string file, Accelerator accelerator)
    {
        Bitmap image = map.ConstructBitmap(accelerator);
        try
        {
            image.Save(file, System.Drawing.Imaging.ImageFormat.Png);
        }
        catch (System.Exception e)
        {
            Console.WriteLine("Error occured when trying to save image: " + file + "\n" + e.ToString());
        }
    }

    protected static void StopWatch(Action func, string processName)
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        func();
        watch.Stop();
        var elapsedMs = watch.ElapsedMilliseconds;
        if (PRINTSTOPWATCH)
            Console.WriteLine($"Time: {elapsedMs / 1000f:F3} s {processName}");
    }

    protected static T[][] TransposeArray<T>(T[][] array)
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

    protected static T[,] TransposeArray<T>(T[,] array)
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