using ConvolutionalNeuralNetwork.Layers;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    [Serializable]
    public class Weights
    {
        [JsonProperty] private ColorVector _weights;
        [JsonProperty] private Color[] _filter;
        private ColorVector _gradient;
        [JsonProperty] private readonly Color[] _firstMoment;
        [JsonProperty] private readonly Color[] _secondMoment;

        public Color this[int index]
        {
            get => _weights[index];
        }

        public void SetWeights(int index, Color color)
        {
            _weights[index] = color;
        }

        public void SetGradient(int index, Color color, float learningRate, float firstMomentDecay, float secondMomentDecay, float clip = 0.5f)
        {
            _gradient[index] = color;

            UpdateWeightsAtIndex(index, learningRate, firstMomentDecay, secondMomentDecay, clip);
        }

        [JsonIgnore] public int Length => _weights.Length;

        public Weights(int length, float mean, float stdDev)
        {
            _weights = new ColorVector(length);

            for(int i = 0; i < length; i++)
            {
                _weights[i] = Color.RandomGauss(mean, stdDev);
            }

            _gradient = new ColorVector(length);
            _firstMoment = new Color[length];
            _secondMoment = new Color[length];
        }

        public Weights(int length, Color color)
        {
            _weights = new ColorVector(length);

            for (int i = 0; i < length; i++)
            {
                _weights[i] = color;
            }

            _gradient = new ColorVector(length);
            _firstMoment = new Color[length];
            _secondMoment = new Color[length];
        }

        [JsonConstructor] private Weights() { }

        [OnDeserialized]
        public void OnDeserialized(StreamingContext context)
        {
            if(_filter != null)
            {
                _weights = new ColorVector(_filter);
                _filter = null;
            }
            _gradient = new ColorVector(Length);
        }

        public void Reset(float mean, float stdDev)
        {
            for (int i = 0; i < Length; i++)
            {
                _weights[i] = Color.RandomGauss(mean, stdDev);
                _firstMoment[i] = Color.Zero;
                _secondMoment[i] = Color.Zero;
            }
        }

        public void Reset(Color color)
        {
            for (int i = 0; i < Length; i++)
            {
                _weights[i] = color;
                _firstMoment[i] = Color.Zero;
                _secondMoment[i] = Color.Zero;
            }
        }

        public ArrayView<T> GradientGPU<T>() where T : unmanaged
        {
            return _gradient.GetArrayViewZeroed<T>();
        }

        public ArrayView<T> WeightsGPU<T>() where T : unmanaged
        {
            return _weights.GetArrayView<T>();
        }

        public void DisposeWeights(uint batchSize)
        {
            _weights.DecrementLiveCount(batchSize);
        }

        public void DisposeGradient(uint batchSize)
        {
            _gradient.DecrementLiveCount(batchSize);
        }

        /// <summary>
        /// Updates the filter weights along with the first and second moments.
        /// </summary>
        /// <param name="learningRate">The overall learning rate for the layer updates, corrected for the influence of bias in the first and second moments.</param>
        /// <param name="firstMomentDecay">The exponential decay rate for the first moment.</param>
        /// <param name="secondMomentDecay">The exponential decay rate for the second moment.</param>
        /// <param name="clip">The maximum absolute value to clip the gradient to.</param>
        public void UpdateWeights(float learningRate, float firstMomentDecay, float secondMomentDecay, float clip = 0.5f)
        {
            _gradient.SyncCPU();

            for (int i = 0; i < Length; i++)
            {
                UpdateWeightsAtIndex(i, learningRate, firstMomentDecay, secondMomentDecay, clip);
            }
            _weights.UpdateIfAllocated();
        }

        private void UpdateWeightsAtIndex(int index, float learningRate, float firstMomentDecay, float secondMomentDecay, float clip)
        {
            Color gradient = _gradient[index].Clip(clip);
            Color first = _firstMoment[index] = firstMomentDecay * _firstMoment[index] + (1 - firstMomentDecay) * gradient;
            Color second = _secondMoment[index] = secondMomentDecay * _secondMoment[index] + (1 - secondMomentDecay) * Color.Pow(gradient, 2);
            _weights[index] -= learningRate * first / (Color.Pow(second, 0.5f) + Utility.AsymptoteErrorColor);
        }

        public void TestFilterGradient(Layer layer, FeatureMap[,] inputs, FeatureMap output, int outputIndex, IOBuffers buffer)
        {
            int inputDimensions = inputs.GetLength(0);

            for (int i = 0; i < inputDimensions; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        inputs[i, 0][j, k] = (i + 1) * new Color(j, k, j - k);
                    }
                }
            }

            for (int i = 0; i < inputDimensions; i++)
            {
                inputs[i, 0].CopyToBuffer(buffer.InputsColor[i, 0]);
            }

            layer.Forward();
            output.SyncCPU(buffer.OutputsColor[outputIndex, 0]);
            FeatureMap gradient = new(output.Width, output.Length, Color.One);
            gradient.CopyToBuffer(buffer.InGradientsColor[outputIndex, 0]);
            layer.Backwards(1, 1, 1);

            FeatureMap testOutput = new(output.Width, output.Length);

            for (int i = 0; i < inputDimensions; i++)
            {
                inputs[i, 0].CopyToBuffer(buffer.InputsColor[i, 0]);
            }

            float h = 0.001f;

            for (int i = 0; i < Length; i++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Color hColor = k switch
                    {
                        0 => new Color(h, 0, 0),
                        1 => new Color(0, h, 0),
                        2 => new Color(0, 0, h)
                    };
                    _weights[i] += hColor;

                    layer.Forward();
                    testOutput.SyncCPU(buffer.OutputsColor[outputIndex, 0]);

                    float testGradient = 0;
                    for (int i2 = 0; i2 < output.Width; i2++)
                    {
                        for (int j2 = 0; j2 < output.Length; j2++)
                        {
                            for (int k2 = 0; k2 < 3; k2++)
                            {
                                testGradient += (testOutput[i2, j2][k2] - output[i2, j2][k2]) / h;
                            }
                        }
                    }

                    Console.WriteLine($"Expected Gradient: {_gradient[i][k]:f4} \t Test Gradient: {testGradient:f4}");
                    _weights[i] -= hColor;
                }
            }
        }
    }
}
