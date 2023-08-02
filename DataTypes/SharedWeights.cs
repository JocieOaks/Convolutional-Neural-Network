using ConvolutionalNeuralNetwork.Layers.Initializers;
using ConvolutionalNeuralNetwork.Layers.Weighted;
using ILGPU;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using static ConvolutionalNeuralNetwork.Layers.BatchNormalization;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    public class SharedWeights : IWeights
    {
        [JsonProperty] Weights _weights;
        Vector _gradient;

        public IWeightInitializer WeightInitializer { get; init; }

        public bool Initialized { get; private set; } = false;

        public void SetWeights(Weights weights)
        {
            if (!Initialized)
            {
                _weights = weights;
                _gradient = new Vector(weights.Length);
                _weights.SetGradient(_gradient);
                Initialized = true;
            }
        }

        public float this[int index] => _weights[index];

        public int Length => _weights.Length;

        public void DecrementLiveGradient(int decrement = 1)
        {
            _weights.DecrementLiveGradient(decrement);
        }

        public void DecrementLiveWeights(int decrement = 1)
        {
            _weights.DecrementLiveWeights(decrement);
        }

        public ArrayView<T> GradientGPU<T>() where T : unmanaged
        {
            return _gradient.GetArrayView<T>();
        }

        public void Reset(float value)
        {
            _weights.Reset(value);
        }

        public void Reset(float mean, float stdDev)
        {
            _weights.Reset(mean, stdDev);
        }

        public ArrayView<T> WeightsGPU<T>() where T : unmanaged
        {
            return _weights.WeightsGPU<T>();
        }
    }
}
