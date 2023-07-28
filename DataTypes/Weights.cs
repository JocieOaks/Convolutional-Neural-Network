using ConvolutionalNeuralNetwork.Layers;
using ConvolutionalNeuralNetwork.Layers.Initializers;
using ConvolutionalNeuralNetwork.Layers.Weighted;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.IR;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Reflection.Emit;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    [Serializable]
    public class Weights
    {
        [JsonProperty] private readonly float[] _firstMoment;
        [JsonProperty] private readonly float[] _secondMoment;
        private Vector _gradient;
        [JsonProperty] private Vector _weights;

        public Weights(int length, IWeightInitializer initializer, WeightedLayer layer)
        {
            _weights = new Vector(length);

            for (int i = 0; i < length; i++)
            {
                _weights[i] = initializer.GetWeight(layer);
            }

            _gradient = new Vector(length);
            _firstMoment = new float[length];
            _secondMoment = new float[length];
        }

        public Weights(int length)
        {
            _weights = new Vector(length);
            _gradient = new Vector(length);
            _firstMoment = new float[length];
            _secondMoment = new float[length];
        }

        [JsonConstructor] private Weights() { }

        [JsonIgnore] public int Length => _weights.Length;

        public float this[int index]
        {
            get => _weights[index];
        }

        public void DecrementLiveGradient(int decrement = 1)
        {
            _gradient.DecrementLiveCount((uint)decrement);
        }

        public void DecrementLiveWeights(int decrement = 1)
        {
            _weights.DecrementLiveCount((uint)decrement);
        }

        public ArrayView<T> GradientGPU<T>() where T : unmanaged
        {
            return _gradient.GetArrayViewZeroed<T>();
        }

        [OnDeserialized]
        public void OnDeserialized(StreamingContext context)
        {
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

        public void SetGradient(int index, float value, AdamHyperParameters hyperParameters, float clip = 0.1f)
        {
            _gradient[index] = value;

            UpdateWeightsAtIndex(index, hyperParameters, clip);
        }

        public void SetWeights(int index, float color)
        {
            _weights[index] = color;
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

            layer.Forward(batchSize);
            for (int i = 0; i < outputDimensions * batchSize; i++)
            {
                outputs[i].SyncCPU(buffer.Output.SubView(outputShapes.Area * i, outputShapes.Area));
                new FeatureMap(outputs[i].Width, outputs[i].Length, 1).CopyToBuffer(buffer.InGradient.SubView(outputShapes.Area * i, outputShapes.Area));
            }

            layer.Backwards(batchSize);

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
                layer.Forward(batchSize);

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

                if (MathF.Abs(_gradient[i] - testGradient) > Math.Max(0.01, testGradient * 0.001))
                {
                    Console.WriteLine($"Expected Gradient: {_gradient[i]:f4} \t Test Gradient: {testGradient:f4}");
                    Console.ReadLine();
                }
                _weights[i] -= h;
            }

        }

        /// <summary>
        /// Updates the filter weights along with the first and second moments.
        /// </summary>
        /// <param name="learningRate">The overall learning rate for the layer updates, corrected for the influence of bias in the first and second moments.</param>
        /// <param name="firstMomentDecay">The exponential decay rate for the first moment.</param>
        /// <param name="secondMomentDecay">The exponential decay rate for the second moment.</param>
        /// <param name="clip">The maximum absolute value to clip the gradient to.</param>
        public void UpdateWeights(AdamHyperParameters hyperParameters, float clip = 1000f)
        {
            _gradient.SyncCPU();

            for (int i = 0; i < Length; i++)
            {
                UpdateWeightsAtIndex(i, hyperParameters, clip);
            }
            _weights.UpdateIfAllocated();
        }

        public ArrayView<T> WeightsGPU<T>() where T : unmanaged
        {
            return _weights.GetArrayView<T>();
        }
        private void UpdateWeightsAtIndex(int index, AdamHyperParameters hyperParameters, float clip)
        {
            float gradient = Math.Clamp(_gradient[index], -clip, clip);
            if(float.IsNaN(gradient))
            {
                throw new Exception();
            }
            float first = hyperParameters.FirstMomentDecay * _firstMoment[index] + (1 - hyperParameters.FirstMomentDecay) * gradient;
            float second = hyperParameters.SecondMomentDecay * _secondMoment[index] + (1 - hyperParameters.SecondMomentDecay) * MathF.Pow(gradient, 2);
            _firstMoment[index] = first;
            _secondMoment[index] = second;
            float result = hyperParameters.LearningRate * first / (MathF.Pow(second, 0.5f) + Utility.ASYMPTOTEERRORCORRECTION);
            _weights[index] -= result;
        }
    }
}
