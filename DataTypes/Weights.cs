using ConvolutionalNeuralNetwork.GPU;
using ConvolutionalNeuralNetwork.Layers.Initializers;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Runtime.Serialization;
using ConvolutionalNeuralNetwork.Layers;
using ConvolutionalNeuralNetwork.Layers.Serial;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="Weights"/> class stores the weights and moments used for <see cref="Layer"/> filters.
    /// </summary>
    [Serializable]
    public class Weights
    {
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float, float, float, float> s_updateAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float, float, float, float>(UpdateKernel);

        [JsonProperty] private Vector _firstMoment;
        private Vector _gradient;
        [JsonProperty] private float _gradientClip;
        private IWeightInitializer _initializer;
        [JsonProperty] private Vector _secondMoment;
        [JsonProperty] private Vector _weights;
        [JsonProperty] private float _weightsClip;

        /// <summary>
        /// Initializes a new instance of the <see cref="Weights"/> class.
        /// </summary>
        /// <param name="initializer">The <see cref="IWeightInitializer"/> used to set the initial values.</param>
        /// <param name="gradientClip">Value to clip gradients. Default 1000.</param>
        /// <param name="weightsClip">Value to clip weights. Default 1000.</param>
        public Weights(IWeightInitializer initializer, float gradientClip = 1000, float weightsClip = 1000)
        {
            _initializer = initializer;
            _gradientClip = gradientClip;
            _weightsClip = weightsClip;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Weights"/> class. Used for deserialization.
        /// </summary>
        [JsonConstructor] private Weights() { }

        /// <value>The number of weights.</value>
        [JsonIgnore] public int Length => _weights.Length;

        /// <summary>
        /// Indexer for <see cref="Weights"/>.
        /// </summary>
        /// <param name="index">The index of the desired weight.</param>
        /// <returns>Returns the weight as <paramref name="index"/>.</returns>
        public float this[int index] => _weights[index];

        public void DecrementLiveGradient(int decrement = 1)
        {
            _gradient.DecrementLiveCount((uint)decrement);
        }

        public void DecrementLiveWeights(int decrement = 1)
        {
            _weights.DecrementLiveCount((uint)decrement);
        }

        public ArrayView<float> GradientGPU()
        {
            return _gradient.GetArrayViewZeroed();
        }

        public void Initialize(int length, SerialWeighted layer)
        {
            if (_weights != null)
            {
                if (length != _weights.Length)
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

        [OnDeserialized]
        public void OnDeserialized(StreamingContext context)
        {
            _gradient = new Vector(Length);
        }

        /// <summary>
        /// Updates the filter weights along with the first and second moments.
        /// </summary>
        /// <param name="hyperParameters">The <see cref="AdamHyperParameters"/> detailing how the <see cref="Weights"/> should be updated.</param>
        public void UpdateWeights(AdamHyperParameters hyperParameters)
        {
            Index1D index = new(Length);
            s_updateAction(index, _weights.GetArrayView(), _firstMoment.GetArrayView(),
                _secondMoment.GetArrayView(), _gradient.GetArrayView(), hyperParameters.LearningRate,
                hyperParameters.FirstMomentDecay, hyperParameters.SecondMomentDecay, _gradientClip, _weightsClip);
            GPUManager.Accelerator.Synchronize();
            _weights.DecrementLiveCount();
            _firstMoment.DecrementLiveCount();
            _secondMoment.DecrementLiveCount();
            _gradient.DecrementLiveCount();
        }

        public ArrayView<float> WeightsGPU()
        {
            return _weights.GetArrayView();
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

        [OnSerializing]
        private void OnSerializing(StreamingContext context)
        {
            _weights.SyncCPU();
            _firstMoment.SyncCPU();
            _secondMoment.SyncCPU();
        }
    }
}
