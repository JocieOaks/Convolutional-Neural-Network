using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers;
using ConvolutionalNeuralNetwork.Layers.Serial;
using ConvolutionalNeuralNetwork.Layers.Initializers;
using Newtonsoft.Json;
using ConvolutionalNeuralNetwork.Layers.Loss;

namespace ConvolutionalNeuralNetwork.Networks
{
    [Serializable]
    public partial class FILM : Network
    {
        private readonly int _pyramidLayers;
        public FILM(Shape inputShape, int pyramidLayers) : base(new FILMLoss())
        {
            _pyramidLayers = pyramidLayers;
            SerialConv[] featuresConvs = new SerialConv[6];
            SerialConv[] flowConvs = new SerialConv[3];

            for (int i = 0; i < 3; i++)
            {
                featuresConvs[i] = new SerialConv((int)Math.Pow(2, 4 + i), 3, 1, new Weights(GlorotUniform.Instance), null);
                featuresConvs[i + 3] = new SerialConv((int)Math.Pow(2, 5 + i), 3, 1, new Weights(GlorotUniform.Instance), null);
                flowConvs[i] = new SerialConv(i == 0 ? 256 : i == 1 ? 128 : 2, 3, 1, new Weights(GlorotUniform.Instance), null);
            }

            _inputShape = inputShape;

            CreateFeatureExtraction(featuresConvs, out SerialFork[] outputs0, out SerialFork[] i0, inputShape);
            CreateFeatureExtraction(featuresConvs, out SerialFork[] outputs1, out SerialFork[] i1, inputShape);

            CreateFlow(outputs0, outputs1, i0, flowConvs, out SerialFork[] flow0, out SerialFork[] f0);
            CreateFlow(outputs1, outputs0, i1, flowConvs, out SerialFork[] flow1, out SerialFork[] f1);

            CreateFusion(f0, f1, flow0, flow1);
        }

        [JsonConstructor] private FILM() : base(new FILMLoss()){ }

        private void CreateFeatureExtraction(SerialConv[] shared, out SerialFork[] outputs, out SerialFork[] originals, Shape inputShape)
        {
            outputs = new SerialFork[_pyramidLayers];
            originals = new SerialFork[_pyramidLayers];

            List<SerialFork>[] skips = new List<SerialFork>[_pyramidLayers];
            for (int i = 0; i < _pyramidLayers; i++)
            {
                skips[i] = new();
            }

            AddInput(inputShape);
            SerialFork inputFork = AddFork();
            originals[0] = inputFork;

            for (int i = 0; i < _pyramidLayers; i++)
            {
                if (i != 0)
                {
                    AddSkipOut(inputFork);
                    AddAveragePool((int)Math.Pow(2, i));
                    originals[i] = AddFork();
                }
                for (int j = 0; j < 3; j++)
                {
                    if (i + j < _pyramidLayers)
                    {
                        AddSerialLayer(shared[j]);
                        AddActivation(Activation.ReLU);
                        AddBatchNormalization();
                        AddSerialLayer(shared[j + 3]);
                        AddActivation(Activation.ReLU);
                        AddBatchNormalization();
                        skips[i + j].Add(AddFork());
                        AddAveragePool(2);
                    }
                }
            }

            for (int i = 0; i < _pyramidLayers; i++)
            {
                AddSkipOut(skips[i].First());
                foreach (var skip in skips[i].Skip(1))
                {
                    AddConcatenation(skip);
                }
                outputs[i] = AddFork();
            }
        }

        private void CreateFlow(SerialFork[] f1, SerialFork[] f0, SerialFork[] i1, SerialConv[] shared, out SerialFork[] flow, out SerialFork[] f)
        {
            SerialFork[] upsampled = new SerialFork[_pyramidLayers - 1];
            flow = new SerialFork[_pyramidLayers];
            f = new SerialFork[_pyramidLayers];

            AddSkipOut(f1[^1]);
            AddConcatenation(f0[^1]);

            AddSerialLayer(shared[0]);
            AddActivation(Activation.ReLU);

            AddSerialLayer(shared[1]);
            AddActivation(Activation.ReLU);

            AddSerialLayer(shared[2]);
            AddActivation(Activation.ReLU);

            flow[^1] = AddFork();
            AddUpsampling(2);
            upsampled[^1] = AddFork();

            for (int i = _pyramidLayers - 2; i >= 0; i--)
            {
                AddConcatenation(f1[i]);
                AddWarp();
                AddConcatenation(f0[i]);

                if (i < 2)
                {
                    AddConvolution(i == 0 ? 64 : 128, 3, 1, activation: Activation.ReLU);
                    AddBatchNormalization();
                    AddConvolution(i == 0 ? 32 : 64, 3, 1, activation: Activation.ReLU);
                    AddBatchNormalization();
                    AddConvolution(2, 3, 1, activation: Activation.ReLU);
                    AddBatchNormalization();
                }
                else
                {
                    AddSerialLayer(shared[0]);
                    AddActivation(Activation.ReLU);
                    AddBatchNormalization();

                    AddSerialLayer(shared[1]);
                    AddActivation(Activation.ReLU);
                    AddBatchNormalization();

                    AddSerialLayer(shared[2]);
                    AddActivation(Activation.ReLU);
                    AddBatchNormalization();
                }
                AddConcatenation(upsampled[i]);

                AddSummation(2);

                flow[i] = AddFork();

                if (i != 0)
                {
                    AddUpsampling(2);
                    upsampled[i - 1] = AddFork();
                }
            }

            for (int i = 0; i < _pyramidLayers; i++)
            {
                AddSkipOut(flow[i]);
                AddConcatenation(f1[i]);
                AddConcatenation(i1[i]);
                AddWarp();
                f[i] = AddFork();
            }
        }

        private void CreateFusion(SerialFork[] f0, SerialFork[] f1, SerialFork[] flow0, SerialFork[] flow1)
        {
            AddSkipOut(f0[^1]);
            AddConcatenation(f1[^1]);
            AddConcatenation(flow0[^1]);
            AddConcatenation(flow1[^1]);


            for (int i = _pyramidLayers - 2; i >= 0; i--)
            {
                AddUpsampling(2);
                AddConvolution(16, 2, 1, activation: Activation.ReLU);
                AddBatchNormalization();

                AddConcatenation(f0[i]);
                AddConcatenation(f1[i]);
                AddConcatenation(flow0[i]);
                AddConcatenation(flow1[i]);

                AddConvolution(16, 3, 1, activation: Activation.ReLU);
                AddBatchNormalization();
                AddConvolution(16, 3, 1, activation: Activation.ReLU);
                AddBatchNormalization();
            }

            AddConvolution(_inputShape.Dimensions, 1, 1, useBias: false);
        }

        public override void StartUp(int maxBatchSize, AdamHyperParameters hyperParameters)
        {
            base.StartUp(maxBatchSize, hyperParameters);

            _outputs = new FeatureMap[maxBatchSize][];
            for (int i = 0; i < maxBatchSize; i++)
            {
                _outputs[i] = new FeatureMap[_inputShape.Dimensions];
                for (int j = 0; j < _inputShape.Dimensions; j++)
                {
                    _outputs[i][j] = new FeatureMap(_inputShape.Width, _inputShape.Length);
                }
            }
        }

        public static FILM LoadFromFile(string file)
        {
            FILM film = null;

            if (File.Exists(file))
            {
                try
                {
                    using (StreamReader r = new(file))
                    {
                        using (JsonReader reader = new JsonTextReader(r))
                        {
                            JsonSerializer serializer = new();
                            serializer.TypeNameHandling = TypeNameHandling.Auto;
                            film = serializer.Deserialize<FILM>(reader);
                        }
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine("Error occured when trying to load data from file: " + file + "\n" + e.ToString());
                }
            }

            return film;
        }


        private float CalculateLoss(Vector[] actual)
        {
            for (int i = 0; i < actual.Length; i++)
            {
                for (int j = 0; j < _inputShape.Dimensions; j++)
                {
                    _outputs[i][j].SyncCPU(Output.SubView((i * _inputShape.Dimensions + j) * _inputShape.Area, _inputShape.Area));
                }
            }

            (float loss, FeatureMap[][] gradients) = GetLoss(_outputs, actual);

            for (int i = 0; i < actual.Length; i++)
            {
                for (int j = 0; j < _inputShape.Dimensions; j++)
                {
                    gradients[i][j].CopyToBuffer(InGradient.SubView((i * _inputShape.Dimensions + j) * _inputShape.Area, _inputShape.Area));
                }
            }

            return loss;
        }

        private (float, FeatureMap[][]) GetLoss(FeatureMap[][] expected, Vector[] actual)
        {
            int area = expected[0][0].Area;
            int width = expected[0][0].Width;

            FeatureMap[][] gradient = new FeatureMap[expected.Length][];
            float loss = 0;
            for(int i = 0; i < expected.Length; i++)
            {
                gradient[i] = new FeatureMap[expected[i].Length];
                for(int j =0; j < expected[i].Length; j++)
                {
                    gradient[i][j] = new FeatureMap(_inputShape);
                    for(int y = 0; y < expected[i][j].Length; y++)
                    {
                        for(int x = 0; x < expected[i][j].Length; x++)
                        {
                            float defect = expected[i][j][x, y] - actual[i][j * area + y * width + x];
                            gradient[i][j][x, y] = defect;
                            loss += MathF.Abs(defect);
                        }
                    }
                }
            }
            return (loss / expected.Length, gradient);
        }

    }
}
