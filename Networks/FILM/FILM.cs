using ConvolutionalNeuralNetwork.DataTypes;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Networks
{
    [Serializable]
    public partial class FILM : Network
    {
        private const int PYRAMIDLAYERS = 5;

        /// <summary>
        /// Loads a <see cref="FILM"/> from a json file.
        /// </summary>
        /// <param name="file">The path of the json file.</param>
        /// <returns>Returns the deserialized <see cref="FILM"/>.</returns>
        public static FILM LoadFromFile(string file)
        {
            FeatureExtraction features0 = FeatureExtraction.LoadFromFile(Path.Combine(file, "Features 0.json"));
            if(features0 == null)
            {
                return null;
            }
            FeatureExtraction features1 = FeatureExtraction.LoadFromFile(Path.Combine(file, "Features 1.json"));
            if(features1 == null)
            {
                return null;
            }
            Flow flow0 = Flow.LoadFromFile(Path.Combine(file, "Flow 0.json"));
            if (flow0 == null)
            {
                return null;
            }
            Flow flow1 = Flow.LoadFromFile(Path.Combine(file, "Flow 1.json"));
            if (flow1 == null)
            {
                return null;
            }
            Fusion fusion = Fusion.LoadFromFile(Path.Combine(file, "Fusion.json"));
            if (fusion == null)
            {
                return null;
            }

            FILM film = new(features0, features1, flow0, flow1, fusion);
            return film;
        }

        private readonly FeatureExtraction _features0;
        private readonly FeatureExtraction _features1;
        private readonly Flow _flow0;
        private readonly Flow _flow1;
        private readonly Fusion _fusion;

        private FeatureMap[][] _outputs;

        private Shape _inputShape;

        public FILM(int pyramidLayers)
        {
            _features0 = new FeatureExtraction(0);
            _features1 = new FeatureExtraction(1);
            _flow0 = new Flow(_features0.OutputLayers, _features1.OutputLayers);
            _flow1 = new Flow(_features1.OutputLayers, _features0.OutputLayers);
            _fusion = new Fusion(_flow0.F, _flow1.F, _flow0.Flows, _flow1.Flows);
        }

        private FILM(FeatureExtraction features0, FeatureExtraction features1, Flow flow0, Flow flow1, Fusion fusion)
        {
            _features0 = features0;
            _features1 = features1;
            _flow0 = flow0;
            _flow1 = flow1;
            _fusion = fusion;
        }

        public override void SaveToFile(string file)
        {
            Directory.CreateDirectory(file);
            _features0.SaveToFile(Path.Combine(file, "Features 0.json"));
            _features1.SaveToFile(Path.Combine(file, "Features 1.json"));
            _flow0.SaveToFile(Path.Combine(file, "Flow 0.json"));
            _flow1.SaveToFile(Path.Combine(file, "Flow 1.json"));
            _fusion.SaveToFile(Path.Combine(file, "Fusion.json"));
        }

        public override void StartUp(int maxBatchSize, int inputWidth, int inputLength, int boolLabels, int floatLabels, AdamHyperParameters hyperParameters, int inputChannels)
        {
            base.StartUp(maxBatchSize, inputWidth, inputLength, boolLabels, floatLabels, hyperParameters.Copy(), inputChannels);
            _startBuffers = new();
            _middleBuffers = new();
            _endBuffers = _startBuffers;

            _features0.SetStartBuffers(this);
            _features1.SetStartBuffers(this);
            _flow0.SetStartBuffers(this);
            _flow1.SetStartBuffers(this);
            _fusion.SetStartBuffers(this);

            _features0.StartUp(maxBatchSize, inputWidth, inputLength, boolLabels, floatLabels, hyperParameters.Copy(), inputChannels);
            _features1.StartUp(maxBatchSize, inputWidth, inputLength, boolLabels, floatLabels, hyperParameters.Copy(), inputChannels);
            _flow0.StartUp(maxBatchSize, inputWidth, inputLength, boolLabels, floatLabels, hyperParameters.Copy(), inputChannels);
            _flow1.StartUp(maxBatchSize, inputWidth, inputLength, boolLabels, floatLabels, hyperParameters.Copy(), inputChannels);
            _fusion.StartUp(maxBatchSize, inputWidth, inputLength, boolLabels, floatLabels, hyperParameters.Copy(), inputChannels);

            _inputShape = new Shape(inputWidth, inputLength, inputChannels);

            _outputs = new FeatureMap[maxBatchSize][];
            for(int i = 0; i < maxBatchSize; i++)
            {
                _outputs[i] = new FeatureMap[inputChannels];
                for(int j = 0; j < inputChannels; j++)
                {
                    _outputs[i][j] = new FeatureMap(inputWidth, inputLength);
                }
            }

            _startBuffers.Allocate(maxBatchSize);
            _middleBuffers.Allocate(maxBatchSize);
            IOBuffers.SetCompliment(_startBuffers, _middleBuffers);
        }

        public float Train(FeatureMap[][] i0, FeatureMap[][] i1, FeatureMap[][] iT)
        {
            int batchSize = i0.Length;

            _features0.Forward(i0);
            _features1.Forward(i1);
            _flow0.Forward(batchSize);
            _flow1.Forward(batchSize);
            _fusion.Forward(batchSize);

            for(int i = 0; i < batchSize; i++)
            {
                for(int j = 0; j < _inputChannels; j++)
                {
                    _outputs[i][j].SyncCPU(_fusion.Output.SubView((i * _inputChannels + j) * _inputShape.Area, _inputShape.Area));
                }
            }

            (float loss, FeatureMap[][] gradients) = GetLoss(_outputs, iT);

            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < _inputChannels; j++)
                {
                    gradients[i][j].CopyToBuffer(_fusion.InGradient.SubView((i * _inputChannels + j) * _inputShape.Area, _inputShape.Area));
                }
            }

            _fusion.Backwards(batchSize);
            _flow1.Backwards(batchSize);
            _flow0.Backwards(batchSize);
            _features1.Backwards(batchSize);
            _features0.Backwards(batchSize);

            return loss;
        }

        public (FeatureMap[][], float) Test(FeatureMap[][] i0, FeatureMap[][] i1, FeatureMap[][] iT)
        {
            int batchSize = i0.Length;

            _features0.Forward(i0);
            _features1.Forward(i1);
            _flow0.Forward(batchSize);
            _flow1.Forward(batchSize);
            _fusion.Forward(batchSize);

            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < _inputChannels; j++)
                {
                    _outputs[i][j].SyncCPU(_fusion.Output.SubView((i * _inputChannels + j) * _inputShape.Area, _inputShape.Area));
                }
            }

            (float loss, FeatureMap[][] _) = GetLoss(_outputs, iT);

            return (_outputs, loss);
        }

        private (float, FeatureMap[][]) GetLoss(FeatureMap[][] expected, FeatureMap[][] actual)
        {

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
                            float defect = expected[i][j][x, y] - actual[i][j][x, y];
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
