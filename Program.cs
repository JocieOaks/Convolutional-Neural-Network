// See https://aka.ms/new-console-template for more information
using System.Diagnostics;
using System.Drawing;
using System.IO;
using Newtonsoft.Json;

if (OperatingSystem.IsWindows())
{
    string directory = "C:\\Users\\joaks\\AppData\\LocalLow\\DefaultCompany\\Models";
    string filename = "model.json";

    Bitmap[] images = new Bitmap[] {
    new Bitmap("C:\\Users\\joaks\\Documents\\Projects\\CLIP C#\\Images\\Athemar.png"),
    new Bitmap("C:\\Users\\joaks\\Documents\\Projects\\CLIP C#\\Images\\Incubus.png"),
    new Bitmap("C:\\Users\\joaks\\Documents\\Projects\\CLIP C#\\Images\\Queltocn.png"),
    new Bitmap("C:\\Users\\joaks\\Documents\\Projects\\CLIP C#\\Images\\Soren.png")
    };

    int[][] vectors = new int[][]
    {
        new int[]{1, 1, 1, 1, 0, 1, 1, 1, 0, 0},
        new int[]{2, 2, 2, 2, 0, 1, 0, 2, 0, 0},
        new int[]{3, 3, 1, 4, 0, 1, 1, 1, 1, 1},
        new int[]{4, 2, 1, 3, 1, 0, 1, 2, 0, 0}
    };

    (FeatureMap, int[])[] input = new (FeatureMap, int[])[images.Length];

    int width = images.Max(x => x.Width);
    int height = images.Max(y => y.Height);


    for (int k = 0; k < images.Length; k++)
    {
        Bitmap image = images[k];
        FeatureMap imageArray = new FeatureMap(width, height);
        int paddingX = (width - image.Width) / 2;
        int paddingY = (height - image.Height) / 2;

        for (int i = 0; i < image.Width; i++)
        {
            for (int j = 0; j < image.Height; j++)
            {
                imageArray[i + paddingX, j + paddingY] = new Color(image.GetPixel(i, j));
            }
        }
        input[k] = (imageArray, vectors[k]);
    }

    CLIP? clip = null;

    string filepath = Path.Combine(directory, filename);

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
        catch(Exception e)
        {
            Console.WriteLine("Error occured when trying to load data from file: " + filepath + "\n" + e.ToString());
        }
    }
    
    clip ??= new(12, 8, 10, input.Length, input[0].Item2.Length);

    float loss = clip.Train(input, 0.01f);
    float initialLoss = loss;
    Print(clip.Score());
    for(int i = 0; i < 100; i++)
    {
        loss = clip.Train(input, 0.05f);
        Console.WriteLine(MathF.Round(loss,5));
        Print(clip.Score());
        Console.WriteLine();
    }

    Console.WriteLine("Initial Loss: " + MathF.Round(initialLoss, 5));
    Console.WriteLine("Final Loss: " + MathF.Round(loss, 5));

    try
    {
        // create the directory the file will be written to if it doesn't already exist
        Directory.CreateDirectory(Path.GetDirectoryName(filepath));

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