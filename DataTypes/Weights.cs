using ConvolutionalNeuralNetwork.Layers;
using ILGPU;
using ILGPU.IR;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    [Serializable]
    public class Weights
    {
        [JsonProperty] private Vector _weights;
        [JsonProperty] private float[] _filter;
        private Vector _gradient;
        [JsonProperty] private readonly float[] _firstMoment;
        [JsonProperty] private readonly float[] _secondMoment;

        public float this[int index]
        {
            get => _weights[index];
        }

        public void SetWeights(int index, float color)
        {
            _weights[index] = color;
        }

        public void SetGradient(int index, float color, float learningRate, float firstMomentDecay, float secondMomentDecay, float clip = 0.1f)
        {
            _gradient[index] = color;

            UpdateWeightsAtIndex(index, learningRate, firstMomentDecay, secondMomentDecay, clip);
        }

        [JsonIgnore] public int Length => _weights.Length;

        public Weights(int length, float mean, float stdDev)
        {
            _weights = new Vector(length);

            for(int i = 0; i < length; i++)
            {
                _weights[i] = Utility.RandomGauss(mean, stdDev);
            }

            _gradient = new Vector(length);
            _firstMoment = new float[length];
            _secondMoment = new float[length];
        }

        public Weights(int length, float value)
        {
            _weights = new Vector(length);

            for (int i = 0; i < length; i++)
            {
                _weights[i] = value;
            }

            _gradient = new Vector(length);
            _firstMoment = new float[length];
            _secondMoment = new float[length];
        }

        public Weights(int length, float limit, bool _)
        {
            _weights = new Vector(length);
            for (int i = 0; i < length; i++)
            {
                _weights[i] = (float)(Utility.Random.NextDouble() * 2 - 1) * limit;
            }

            _gradient = new Vector(length);
            _firstMoment = new float[length];
            _secondMoment = new float[length];
        }

        [JsonConstructor] private Weights() { }

        [OnDeserialized]
        public void OnDeserialized(StreamingContext context)
        {
            if(_filter != null)
            {
                _weights = new Vector(_filter);
                _filter = null;
            }
            _gradient = new Vector(Length);
        }

        public void Reset(float mean, float stdDev)
        {
            for (int i = 0; i < Length; i++)
            {
                _weights[i] = Utility.RandomGauss(mean, stdDev);
                _firstMoment[i] = 0;
                _secondMoment[i] = 0;
            }
        }

        public void Reset(float value)
        {
            for (int i = 0; i < Length; i++)
            {
                _weights[i] = value;
                _firstMoment[i] = 0;
                _secondMoment[i] = 0;
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

        public void DecrementLiveWeights(int decrement = 1)
        {
            _weights.DecrementLiveCount((uint)decrement);
        }

        public void DecrementLiveGradient(int decrement = 1)
        {
            _gradient.DecrementLiveCount((uint)decrement);
        }

        /// <summary>
        /// Updates the filter weights along with the first and second moments.
        /// </summary>
        /// <param name="learningRate">The overall learning rate for the layer updates, corrected for the influence of bias in the first and second moments.</param>
        /// <param name="firstMomentDecay">The exponential decay rate for the first moment.</param>
        /// <param name="secondMomentDecay">The exponential decay rate for the second moment.</param>
        /// <param name="clip">The maximum absolute value to clip the gradient to.</param>
        public void UpdateWeights(float learningRate, float firstMomentDecay, float secondMomentDecay, float clip = 1000f)
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
            float gradient = Math.Clamp(_gradient[index], -clip, clip);
            float first = firstMomentDecay * _firstMoment[index] + (1 - firstMomentDecay) * gradient;
            float second = secondMomentDecay * _secondMoment[index] + (1 - secondMomentDecay) * MathF.Pow(gradient, 2);
            _firstMoment[index] = first;
            _secondMoment[index] = second;
            float result = learningRate * first / (MathF.Pow(second, 0.5f) + Utility.ASYMPTOTEERRORCORRECTION);
            _weights[index] -= result;
        }

        public void TestFilterGradient(ILayer layer, Shape inputShape, Shape outputShapes, IOBuffers buffer, int batchSize)
        {
            int inputDimensions = inputShape.Dimensions;
            int outputDimensions = outputShapes.Dimensions;
            FeatureMap[] inputs = new FeatureMap[inputDimensions * batchSize];
            for (int i = 0; i < inputDimensions * batchSize; i++)
            {
                inputs[i] = new FeatureMap(inputShape);
                for (int j = 0; j < inputShape.Length; j++)
                {
                    for (int k = 0; k < inputShape.Width; k++)
                    {
                        inputs[i][j, k] = (i + 1) * (j - k);
                    }
                }
            }

            for (int i = 0; i < inputDimensions * batchSize; i++)
            {
                inputs[i].CopyToBuffer(buffer.Input.SubView(inputShape.Area * i, inputShape.Area));
            }

            FeatureMap[] outputs = new FeatureMap[outputDimensions * batchSize];
            for (int i = 0; i < outputDimensions * batchSize; i++)
            {
                outputs[i] = new FeatureMap(outputShapes);
            }

            layer.Forward();
            for (int i = 0; i < outputDimensions * batchSize; i++)
            {
                outputs[i].SyncCPU(buffer.Output.SubView(outputShapes.Area * i, outputShapes.Area));
                new FeatureMap(outputs[i].Width, outputs[i].Length, 1).CopyToBuffer(buffer.InGradient.SubView(outputShapes.Area * i, outputShapes.Area));
            }

            layer.Backwards(1, 1, 1);

            FeatureMap[] testOutput = new FeatureMap[outputDimensions * batchSize];
            for (int i = 0; i < outputDimensions * batchSize; i++)
            {
                testOutput[i] = new(outputs[i].Width, outputs[i].Length);
            }

            for (int i = 0; i < inputDimensions * batchSize; i++)
            {
                inputs[i].CopyToBuffer(buffer.Input.SubView(inputShape.Area * i, inputShape.Area));
            }

            float h = 0.001f;

            for (int i = 0; i < Length; i++)
            {
                _weights[i] += h;
                _weights.UpdateIfAllocated();
                layer.Forward();
                
                float testGradient = 0;
                for (int i2 = 0; i2 < outputDimensions * batchSize; i2++)
                {
                    testOutput[i2].SyncCPU(buffer.Output.SubView(outputShapes.Area * i2, outputShapes.Area));
                    for (int j2 = 0; j2 < testOutput[i2].Width; j2++)
                    {
                        for (int k2 = 0; k2 < testOutput[i2].Length; k2++)
                        {

                            testGradient += (testOutput[i2][j2, k2] - outputs[i2][j2, k2]) / h;

                        }
                    }
                }

                Console.WriteLine($"Expected Gradient: {_gradient[i]:f4} \t Test Gradient: {testGradient:f4}");
                _weights[i] -= h;
            }
            
        }
    }
}
