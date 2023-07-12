﻿using ConvolutionalNeuralNetwork.Layers;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    [Serializable]
    public class Filter
    {
        [JsonProperty] private readonly Color[] _filter;
        private Color[] _gradient;
        [JsonProperty] private readonly Color[] _firstMoment;
        [JsonProperty] private readonly Color[] _secondMoment;

        private MemoryBuffer1D<Color, Stride1D.Dense> _deviceGradient;
        private MemoryBuffer1D<Color, Stride1D.Dense> _deviceFilter;

        public Color this[int index]
        {
            get => _filter[index];
        }

        public void SetFilter(int index, Color color)
        {
            _filter[index] = color;
        }

        public void SetGradient(int index, Color color, float learningRate, float firstMomentDecay, float secondMomentDecay, float clip = 0.5f)
        {
            _gradient[index] = color;

            UpdateFilterAtIndex(index, learningRate, firstMomentDecay, secondMomentDecay, clip);
        }

        [JsonIgnore] public int Length => _filter.Length;

        public Filter(int length, float mean, float stdDev)
        {
            _filter = new Color[length];

            for(int i = 0; i < length; i++)
            {
                _filter[i] = Color.RandomGauss(mean, stdDev);
            }

            _gradient = new Color[length];
            _firstMoment = new Color[length];
            _secondMoment = new Color[length];
        }

        public Filter(int length, Color color)
        {
            _filter = new Color[length];

            for (int i = 0; i < length; i++)
            {
                _filter[i] = color;
            }

            _gradient = new Color[length];
            _firstMoment = new Color[length];
            _secondMoment = new Color[length];
        }

        [JsonConstructor] private Filter() { }

        [OnDeserialized]
        public void OnDeserialized(StreamingContext context)
        {
            _gradient = new Color[Length];
        }

        public void Reset(float mean, float stdDev)
        {
            for (int i = 0; i < Length; i++)
            {
                _filter[i] = Color.RandomGauss(mean, stdDev);
                _firstMoment[i] = Color.Zero;
                _secondMoment[i] = Color.Zero;
            }
        }

        public void Reset(Color color)
        {
            for (int i = 0; i < Length; i++)
            {
                _filter[i] = color;
                _firstMoment[i] = Color.Zero;
                _secondMoment[i] = Color.Zero;
            }
        }

        public ArrayView<float> GradientFloatGPU()
        {
            SetDeviceGradient();
            return new ArrayView<float>(_deviceGradient, 0, 3 * Length);
        }

        public ArrayView<Color> GradientColorGPU()
        {
            SetDeviceGradient();
            return new ArrayView<Color>(_deviceGradient, 0, Length);
        }

        private void SetDeviceGradient()
        {
            if (_deviceGradient != null && !_deviceGradient.IsDisposed)
            {
                _deviceGradient.Dispose();
            }

            _deviceGradient = Utility.Accelerator.Allocate1D<Color>(Length);
            _deviceGradient.MemSetToZero();
        }

        public void DisposeGradient()
        {
            if(_deviceGradient != null && !_deviceGradient.IsDisposed)
            {
                _deviceGradient.CopyToCPU(_gradient);
            }
            _deviceGradient?.Dispose();
        }

        public ArrayView<Color> FilterColorGPU()
        {
            if (_deviceFilter != null && !_deviceFilter.IsDisposed)
            {
                _deviceFilter.Dispose();
            }

            _deviceFilter = Utility.Accelerator.Allocate1D(_filter);
            return new ArrayView<Color>(_deviceFilter, 0, Length);
        }

        public ArrayView<float> FilterFloatGPU()
        {
            if (_deviceFilter != null && !_deviceFilter.IsDisposed)
            {
                _deviceFilter.Dispose();
            }

            _deviceFilter = Utility.Accelerator.Allocate1D(_filter);
            return new ArrayView<float>(_deviceFilter, 0, 3 * Length);
        }

        public void DisposeFilter()
        {
            _deviceFilter?.Dispose();
        }

        /// <summary>
        /// Updates the filter weights along with the first and second moments.
        /// </summary>
        /// <param name="learningRate">The overall learning rate for the layer updates, corrected for the influence of bias in the first and second moments.</param>
        /// <param name="firstMomentDecay">The exponential decay rate for the first moment.</param>
        /// <param name="secondMomentDecay">The exponential decay rate for the second moment.</param>
        /// <param name="clip">The maximum absolute value to clip the gradient to.</param>
        public void UpdateFilter(float learningRate, float firstMomentDecay, float secondMomentDecay, float clip = 0.5f)
        {
            if (_deviceGradient != null && !_deviceGradient.IsDisposed)
            {
                _deviceGradient.CopyToCPU(_gradient);
            }

            for (int i = 0; i < Length; i++)
            {
                UpdateFilterAtIndex(i, learningRate, firstMomentDecay, secondMomentDecay, clip);
            }
        }

        private void UpdateFilterAtIndex(int index, float learningRate, float firstMomentDecay, float secondMomentDecay, float clip)
        {
            Color gradient = _gradient[index].Clip(clip);
            Color first = _firstMoment[index] = firstMomentDecay * _firstMoment[index] + (1 - firstMomentDecay) * gradient;
            Color second = _secondMoment[index] = secondMomentDecay * _secondMoment[index] + (1 - secondMomentDecay) * Color.Pow(gradient, 2);
            _filter[index] -= learningRate * first / (Color.Pow(second, 0.5f) + Utility.AsymptoteErrorColor);
        }

        public void TestFilterGradient(Layer layer, FeatureMap input, FeatureMap output, IOBuffers buffer)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    input[i, j] = new Color(i, j, i - j);
                }
            }
            input.CopyToBuffer(buffer.InputsColor[0, 0]);

            layer.Forward();
            output.CopyFromBuffer(buffer.OutputsColor[0, 0]);
            FeatureMap gradient = new(output.Width, output.Length, Color.One);
            gradient[0, 0] = Color.One;
            gradient.CopyToBuffer(buffer.InGradientsColor[0, 0]);
            layer.Backwards(1, 1, 1);
            FeatureMap outGradient = new(3, 3);
            outGradient.CopyFromBuffer(buffer.OutGradientsColor[0, 0]);

            FeatureMap testOutput = new(output.Width, output.Length);

            input.CopyToBuffer(buffer.InputsColor[0, 0]);

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
                    _filter[i] += hColor;

                    layer.Forward();
                    testOutput.CopyFromBuffer(buffer.OutputsColor[0, 0]);

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
                    _filter[i] -= hColor;
                }
            }
        }
    }
}
