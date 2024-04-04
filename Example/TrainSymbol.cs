using System.Drawing;
using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Example
{
    /// <summary>
    /// The <see cref="TrainSymbol"/> class is a helper class for creating and training a <see cref="SymbolGAN"/> for a Windows platform.
    /// </summary>
    internal class TrainSymbol
    {
        private readonly List<Tensor> _trainingData = new();
        private string _imagesFolder;
        private TensorShape? _tensorShape;

        /// <summary>
        /// Creates and trains a new <see cref="SymbolGAN"/>.
        /// </summary>
        public void Start()
        {
            if (!OperatingSystem.IsWindows()) return;

            Console.WriteLine("Enter Image Directory");
            string directory = Console.ReadLine();
             while (!Directory.Exists(directory))
             {
                 Console.WriteLine("Directory cannot be found.\nEnter Image Directory");
                 directory = Console.ReadLine();
             }

            _imagesFolder = directory;

            //Training images are imported on a separate thread so that training can begin immediately.
            ThreadStart numbersThreadDelegate = GetData;
            Thread sequenceThread = new(numbersThreadDelegate);
            sequenceThread.Start();
            
            Console.WriteLine("Enter Generated Output Directory");
            string outputDirectory = Console.ReadLine();
             while (string.IsNullOrEmpty(outputDirectory))
             {
                 outputDirectory = Console.ReadLine();
             }
            if (Directory.Exists(outputDirectory))
            {
                Directory.Delete(outputDirectory, true);
            }
            Directory.CreateDirectory(outputDirectory);

            Console.WriteLine("Enter Batch Size");
            int batchSize;
            while (!int.TryParse(Console.ReadLine(), out batchSize) || batchSize < 1)
            {
                Console.WriteLine("Invalid Batch Size.\nEnter Batch Size");
            }

            Console.WriteLine("Enter Latent Dimensions");
            int latentDimensions;
            while (!int.TryParse(Console.ReadLine(), out latentDimensions) || latentDimensions < 1)
            {
                Console.WriteLine("Invalid Dimensions.\nEnter Latent Dimensions");
            }

            Console.WriteLine("Enter Number of Epochs");
            int epochs;
            while (!int.TryParse(Console.ReadLine(), out epochs) || epochs < 1)
            {
                Console.WriteLine("Invalid Value.\nEnter Number of Epochs");
            }
            
            Train(outputDirectory, batchSize, latentDimensions, epochs);
        }

        private void GetData()
        {
            if (!OperatingSystem.IsWindows()) return;
            _tensorShape = null;

            foreach (var file in Directory.EnumerateFiles(_imagesFolder, "*.png"))
            {
                try
                {
                    Bitmap bitmap = new(file);

                    Tensor image = TensorUtility.BitmapToTensor(bitmap, ImageChannels.Greyscale);

                    if (!_tensorShape.HasValue)
                    {
                        _tensorShape = image.Shape;
                    }
                    else
                    {
                        if (_tensorShape.Value != image.Shape)
                            throw new Exception("Images are not of equal size.");

                    }

                    _trainingData.Add(image);

                }
                catch (Exception e)
                {
                    throw new Exception("Error occurred when trying to load data from file: " + file + "\n" + e);
                }

            }
        }

        private void Train(string outputDirectory, int batchSize, int latentDimensions, int epochs)
        {
            if (!OperatingSystem.IsWindows()) return;
            while (_trainingData.Count == 0 || _tensorShape == null)
            {
                Thread.Sleep(100);
            }

            SymbolGAN gan = new(_tensorShape.Value, batchSize, latentDimensions);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                Console.WriteLine($"\nEpoch: {epoch}");
                gan.Train(_trainingData);
                Directory.CreateDirectory(Path.Combine(outputDirectory, $"Epoch {epoch}"));

                Tensor[] output = gan.Test();

                for (int j = 0; j < output.Length; j++)
                {
                    Bitmap generatedBitmap = TensorUtility.TensorToBitmap(output[j]);
                    generatedBitmap.Save(Path.Combine(outputDirectory, $"Epoch {epoch}\\ Image {j}.png"));
                }
            }
        }
    }
}
