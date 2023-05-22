// See https://aka.ms/new-console-template for more information
using System.Diagnostics;
using System.Drawing;
using System.IO;
using Newtonsoft.Json;

if (OperatingSystem.IsWindows())
{
    string directory = "C:\\Users\\joaks\\AppData\\LocalLow\\DefaultCompany\\CLIP";
    string model = "Models\\Model 1.json";
    List<ClassifiedImage> classifications = new();

    foreach(var file in Directory.EnumerateFiles(directory, "Classified*.json"))
    {
        try
        {
            // load the serialized data from the file
            string dataToLoad = "";
            using (FileStream stream = new(file, FileMode.Open))
            {
                using (StreamReader reader = new(stream))
                {
                    dataToLoad = reader.ReadToEnd();
                }
            }

            // deserialize the data from Json back into the C# object
            List<ClassifiedImage>? newClassifications = JsonConvert.DeserializeObject<List<ClassifiedImage>>(dataToLoad);
            if(newClassifications != null)
                classifications.AddRange(newClassifications);
        }
        catch (System.Exception e)
        {
            throw new Exception("Error occured when trying to load data from file: " + file + "\n" + e);
        }
    }
    List<ImageData> images = new();
    foreach(var file in Directory.EnumerateFiles(directory, "Images*.json"))
    {
        try
        {
            using(StreamReader r = new StreamReader(file))
            {
                using(JsonReader reader = new JsonTextReader(r))
                {
                    JsonSerializer serializer = new JsonSerializer();
                    List<ImageData>? data = serializer.Deserialize<List<ImageData>>(reader);
                    if(data != null)
                        images.AddRange(data);
                }
            }
        }
        catch (System.Exception e)
        {
            throw new Exception("Error occured when trying to load data from file: " + file + "\n" + e);
        }
    }

    if (classifications.Count > 0 && images.Count > 0)
    {
        (FeatureMap image, bool[] bools, float[] floats)[] input = new (FeatureMap, bool[], float[])[classifications.Count];
        for (int i = 0; i < classifications.Count; i++)
        {
            (input[i].bools, input[i].floats) = classifications[i].ToVector(3, 3, 8, 3, 3, 20);
            input[i].image = images.First(x => x.ImageName == classifications[i].ImageName).ToFeatureMap();
        }


        CLIP? clip = null;

        string filepath = Path.Combine(directory, model);

        if (File.Exists(filepath))
        {
            try
            {
                string dataToLoad = "";
                using (FileStream stream = new(filepath, FileMode.Open))
                {
                    using (StreamReader read = new(stream))
                    {
                        dataToLoad = read.ReadToEnd();
                    }
                }
                clip = JsonConvert.DeserializeObject<CLIP>(dataToLoad, new JsonSerializerSettings
                {
                    TypeNameHandling = TypeNameHandling.Auto
                });
            }
            catch (Exception e)
            {
                Console.WriteLine("Error occured when trying to load data from file: " + filepath + "\n" + e.ToString());
            }
        }

        Console.WriteLine("Data Loaded");

        (FeatureMap image, bool[] bools, float[] floats)[] testingData = input.TakeLast(8).ToArray();
        (FeatureMap image, bool[] bools, float[] floats)[] trainingData = input.SkipLast(8).ToArray();
        clip ??= new(12, 16, 16, 8, input[0].bools.Length, input[0].floats.Length, input[0].image.Width, input[0].image.Length);
        float[] epochLoss = new float[100];
        float[] epochAccuracy = new float[100];
        for (int epoch = 1; epoch <= 100; epoch++)
        {
            for (int i = 0; i < trainingData.Length; i++)
            {
                int n = CLIP.Random.Next(i, trainingData.Length);
                (trainingData[i], trainingData[n]) = (trainingData[n], trainingData[i]);
            }



            float loss = 0;
            for (int i = 0; i < trainingData.Length / 8; i++)
            {
                loss = clip.Train(trainingData.Skip(i * 8).Take(8).ToArray(), 0.1f);
                Console.WriteLine(MathF.Round(loss, 5));
                Print(clip.Score());
                Console.WriteLine();
            }

            (epochLoss[epoch], epochAccuracy[epoch]) = clip.Test(testingData);

            Console.WriteLine($"Epoch {epoch}");
            if (epoch != 0)
                Console.WriteLine($"Previous Loss: {MathF.Round(epochLoss[epoch - 1], 3)} \t Previous Accuracy: {MathF.Round(epochAccuracy[epoch - 1], 3)}");
            Console.WriteLine($"Final Loss: {MathF.Round(epochLoss[epoch], 3)} \t Final Accuracy: {MathF.Round(epochAccuracy[epoch], 3)}");
            if (epoch % 10 == 9)
                for (int i = 0; i < epoch; i++)
                {
                    Console.WriteLine($"{i + 1}: {MathF.Round(epochLoss[i], 3)}");
                }
        }
        try
        {
            // create the directory the file will be written to if it doesn't already exist
            Directory.CreateDirectory(Path.GetDirectoryName(filepath)!);

            // serialize the C# game data object into Json
            string dataToStore = JsonConvert.SerializeObject(clip, Formatting.Indented, new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.Auto
            });

            // write the serialized data to the file
            using (FileStream stream = File.Create(filepath))
            {
                using (StreamWriter writer = new(stream))
                {
                    writer.Write(dataToStore);
                }
            }

        }
        catch (System.Exception e)
        {
            Console.WriteLine("Error occured when trying to save data to file: " + filepath + "\n" + e.ToString());
        }
    }
}
else
{
    BackPropogationTest test = new BackPropogationTest();
    for (int i = 0; i < 10; i++)
    {
        float loss = test.Test(0.05f, 0.05f);
        Console.WriteLine($"{i} {MathF.Round(loss,4)}");
    }
}

static void Print(float[,] matrix)
{
    for(int i = 0; i < matrix.GetLength(0); i++)
    {
        string text = "";
        for (int j = 0; j < matrix.GetLength(1); j++)
        {
            text += MathF.Round(matrix[i, j], 3) + " ";
        }
        Console.WriteLine(text);
    }
}