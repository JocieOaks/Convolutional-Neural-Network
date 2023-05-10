// See https://aka.ms/new-console-template for more information
using System.Drawing;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.IO;

if (OperatingSystem.IsWindows())
{
    Bitmap[] images = new Bitmap[4];

    images[0] = new Bitmap("C:\\Users\\joaks\\Documents\\Projects\\CLIP C#\\Images\\Athemar.png");
    images[1] = new Bitmap("C:\\Users\\joaks\\Documents\\Projects\\CLIP C#\\Images\\Incubus.png");
    images[2] = new Bitmap("C:\\Users\\joaks\\Documents\\Projects\\CLIP C#\\Images\\Queltocn.png");
    images[3] = new Bitmap("C:\\Users\\joaks\\Documents\\Projects\\CLIP C#\\Images\\Soren.png");

    int[][] vectors = new int[][]
    {
        new int[]{1, 1, 1, 1, 0, 1, 1, 1, 0, 0},
        new int[]{2, 2, 2, 2, 0, 1, 0, 2, 0, 0},
        new int[]{3, 3, 1, 4, 0, 1, 1, 1, 1, 1},
        new int[]{4, 2, 1, 3, 1, 0, 1, 2, 0, 0}
    };

    (Color[,], int[])[] input = new (Color[,], int[])[images.Length];

    int width = images.Max(x => x.Width);
    int height = images.Max(y => y.Height);
    

    for(int k = 0; k < images.Length; k++)
    {
        Bitmap image = images[k];
        Color[,] imageArray = new Color[width, height];
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

    Console.WriteLine("Learning started");
    CLIP clip = new(16, 8, 10, input);
    float[,] matrix = clip.Forward();
    Console.WriteLine(matrix[0,0]);
}