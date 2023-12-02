using ConvolutionalNeuralNetwork.GPU;
using ConvolutionalNeuralNetwork.Layers;
using ConvolutionalNeuralNetwork.Layers.Initializers;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Runtime.Serialization;
using ConvolutionalNeuralNetwork.Layers.Serial;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    [Serializable]
    public class Weights
    {
        [JsonProperty] private Vector _firstMoment;
        [JsonProperty] private Vector _secondMoment;
        private Vector _gradient;
        [JsonProperty] private Vector _weights;
        [JsonProperty] public bool IgnoreClip { private get; init; }
        private IWeightInitializer _initializer;

        public Weights(IWeightInitializer initializer)
        {
            _initializer = initializer;
        }

        [JsonConstructor] private Weights() { }

        [JsonIgnore] public int Length => _weights.Length;

        public float this[int index] => _weights[index];

        public Weights Slice(int index, int length)
        {
            List<float> weights = new List<float>();
            for(int i = index; i < index + length; i++)
            {
                weights.Add(this[i]);
            }

            return new Weights(new Predefined(weights));
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

        public void SetGradient(Vector gradient)
        {
            _gradient = gradient;
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

            if (layer is IUnchangedLayer)
            {
                for (int i = 0; i < outputDimensions * batchSize; i++)
                {
                    outputs[i].SyncCPU(buffer.Input.SubView(outputShapes.Area * i, outputShapes.Area));
                    new FeatureMap(outputs[i].Width, outputs[i].Length, 1).CopyToBuffer(buffer.Gradient.SubView(outputShapes.Area * i, outputShapes.Area));
                }
            }
            else
            {
                for (int i = 0; i < outputDimensions * batchSize; i++)
                {
                    outputs[i].SyncCPU(buffer.Output.SubView(outputShapes.Area * i, outputShapes.Area));
                    new FeatureMap(outputs[i].Width, outputs[i].Length, 1).CopyToBuffer(buffer.InGradient.SubView(outputShapes.Area * i, outputShapes.Area));
                }
            }
            layer.Backwards(batchSize, true);
            _gradient.SyncCPU();

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

                if(layer is IUnchangedLayer)
                {
                    for (int j = 0; j < inputDimensions * batchSize; j++)
                    {
                        inputs[j].CopyToBuffer(buffer.Input.SubView(inputShape.Area * j, inputShape.Area));
                    }
                }

                layer.Forward(batchSize);

                float testGradient = 0;
                for (int i2 = 0; i2 < outputDimensions * batchSize; i2++)
                {
                    testOutput[i2].SyncCPU(layer is IUnchangedLayer
                        ? buffer.Input.SubView(outputShapes.Area * i2, outputShapes.Area)
                        : buffer.Output.SubView(outputShapes.Area * i2, outputShapes.Area));
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
        /// <param name="hyperParameters">The <see cref="AdamHyperParameters"/> detailing how the <see cref="Weights"/> should be updated.</param>
        /// <param name="weightClip">The maximum absolute value to weightClip the gradient to.</param>
        public void UpdateWeights(AdamHyperParameters hyperParameters, float gradientClip, float weightClip)
        {
            Index1D index = new(Length);
            s_updateAction(index, _weights.GetArrayView<float>(), _firstMoment.GetArrayView<float>(),
                _secondMoment.GetArrayView<float>(), _gradient.GetArrayView<float>(), hyperParameters.LearningRate,
                hyperParameters.FirstMomentDecay, hyperParameters.SecondMomentDecay, gradientClip, IgnoreClip ? 1000 : weightClip);
            GPUManager.Accelerator.Synchronize();
            _weights.DecrementLiveCount();
            _firstMoment.DecrementLiveCount();
            _secondMoment.DecrementLiveCount();
            _gradient.DecrementLiveCount();
        }

        public ArrayView<T> WeightsGPU<T>() where T : unmanaged
        {
            return _weights.GetArrayView<T>();
        }

        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float, float, float, float> s_updateAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float, float, float, float>(UpdateKernel);

        [OnSerializing]
        private void OnSerializing(StreamingContext context)
        {
            _weights.SyncCPU();
            _firstMoment.SyncCPU();
            _secondMoment.SyncCPU();
        }

        private static void UpdateKernel(Index1D index, ArrayView<float> weights, ArrayView<float> firstMoment, ArrayView<float> secondMoment, ArrayView<float> gradients, float learningRate, float firstMomentDecay, float secondMomentDecay, float gradientClip, float weightClip)
        {
            float gradient = XMath.Clamp(gradients[index], -gradientClip, gradientClip);
            float first = firstMomentDecay * firstMoment[index] + (1 - firstMomentDecay) * gradient;
            float second = secondMomentDecay * secondMoment[index] + (1 - secondMomentDecay) * MathF.Pow(gradient, 2);
            firstMoment[index] = first;
            secondMoment[index] = second;
            float result = learningRate * first / (XMath.Sqrt(second) + Utility.ASYMPTOTEERRORCORRECTION);
            weights[index] -= result;
            weights[index] = XMath.Clamp(weights[index], -weightClip, weightClip);
        }

        public void Initialize(int length, SerialWeighted layer)
        {
            if(_weights != null)
            {
                if(length != _weights.Length)
                {
                    throw new ArgumentException("Weights are incompatible with layer.");
                }
            }
            else
            {
                _weights = new Vector(length);

                _initializer ??= GlorotUniform.Instance;

                for (int i = 0; i < length; i++)
                {
                    _weights[i] = _initializer.GetWeight(layer);
                }

                _gradient = new Vector(length);
                _firstMoment = new Vector(length);
                _secondMoment = new Vector(length);
            }
        }
    }
}
