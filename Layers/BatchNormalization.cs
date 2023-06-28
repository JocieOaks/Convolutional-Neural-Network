﻿using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using Newtonsoft.Json;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="BatchNormalization"/> class is a <see cref="Layer"/> for normalizing batches of <see cref="FeatureMap"/>s
    /// so that their mean is 0 and standard deviation 1.
    /// </summary>
    [Serializable]
    public class BatchNormalization : Layer, ISecondaryLayer
    {
        [JsonProperty] private ColorVector _bias;
        [JsonProperty] private ColorVector _biasFirstMoment;
        [JsonProperty] private ColorVector _biasSecondMoment;
        private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceGradients;
        private MemoryBuffer1D<StaticLayerInfo, Stride1D.Dense>[] _deviceInfos;
        private MemoryBuffer1D<Color, Stride1D.Dense>[] _deviceMeans;
        private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceSums;
        private MemoryBuffer1D<Color, Stride1D.Dense>[] _deviceValues;
        private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceVariances;
        private ColorVector _mean;
        private ColorVector _sigma;
        [JsonProperty] private ColorVector _weight;
        [JsonProperty] private ColorVector _weightFirstMoment;
        [JsonProperty] private ColorVector _weightSecondMoment;

        /// <summary>
        /// Initializes a new instance of the <see cref="BatchNormalization"/> class.
        /// </summary>
        [JsonConstructor]
        public BatchNormalization() : base(1, 1)
        {
        }

        /// <inheritdoc/>
        [JsonIgnore] public override string Name => "Batch Normalization Layer";

        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            LoadCommon(true);
            GPUGradients();
            GPUBackPropogate(_outGradients);
            DisposeCommon();
            UpdateWeights(learningRate,firstMomentDecay, secondMomentDecay);
        }

        public void GPUGradients()
        {
            Accelerator accelerator = Utility.Accelerator;

            var gradientKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<StaticLayerInfo>>(GradientsKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceGradients[i] = accelerator.Allocate1D<float>(9);
                _deviceGradients[i].MemSetToZero();
                _deviceMeans[i] = accelerator.Allocate1D(new Color[] { _mean[i] });
                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceOutputs[i, j] = _outputs[i, j].Allocate(accelerator);

                    gradientKernal(index, _deviceInputs[i, j].View, _deviceInGradients[i, j].View, _deviceOutputs[i, j].View, _deviceMeans[i].View, _deviceGradients[i].View, _deviceInfos[i].View);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for(int j = 0; j < _batchSize; j++)
                {
                    _deviceOutputs[i, j].Dispose();
                }
                _gradients[i] = new();
                _gradients[i].CopyFromBuffer(_deviceGradients[i]);
                _deviceGradients[i].Dispose();

                _gradients[i].SigmaGradient *= Color.Pow(_sigma[i], -1.5f) * _weight[i] * -0.5f;
                _gradients[i].MeanGradient = -_gradients[i].BiasGradient * _weight[i] / _sigma[i];
                _deviceMeans[i].Dispose();

            }
        }

        public void CPUGradients()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                _gradients[i] = new();
                for (int j = 0; j < _batchSize; j++)
                {
                    for (int y = 0; y < _inputs[i, j].Length; y++)
                    {
                        for (int x = 0; x < _inputs[i, j].Width; x++)
                        {
                            _gradients[i].WeightGradient += _inGradients[i, j][x, y] * _outputs[i, j][x, y];
                            _gradients[i].BiasGradient += _inGradients[i, j][x, y];
                            _gradients[i].SigmaGradient += _inGradients[i, j][x, y] * (_inputs[i, j][x, y] - _mean[i]);
                        }
                    }
                }

                _gradients[i].SigmaGradient *= Color.Pow(_sigma[i], -1.5f) * _weight[i] * -0.5f;
                _gradients[i].MeanGradient = -_gradients[i].BiasGradient * _weight[i] / _sigma[i];
            }
        }

        private Gradients[] _gradients;

        public void GPUBackPropogate(FeatureMap[,] outGradients)
        {
            Accelerator accelerator = Utility.Accelerator;
            accelerator.Synchronize();

            var backwardsKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<Color>, ArrayView<StaticLayerInfo>>(BackwardsKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                float invM = 1f / (Infos(i).Area * _batchSize);

                _deviceValues[i] = accelerator.Allocate1D(new Color[] { _weight[i] / _sigma[i], 2 * invM * _gradients[i].SigmaGradient, _mean[i], invM * _gradients[i].MeanGradient });

                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceOutGradients[i, j] = outGradients[i, j].AllocateFloat(accelerator, false);

                    backwardsKernal(index, _deviceInputs[i, j].View, _deviceInGradients[i, j].View, _deviceOutGradients[i, j].View, _deviceValues[i].View, _deviceInfos[i].View);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    outGradients[i, j].CopyFromBuffer(_deviceOutGradients[i, j]);
                    _deviceOutGradients[i, j].Dispose();
                    
                }
                _deviceValues[i].Dispose();
            }
        }

        public void UpdateWeights(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                Color first = _weightFirstMoment[i] = firstMomentDecay * _weightFirstMoment[i] + (1 - firstMomentDecay) * _gradients[i].WeightGradient;
                Color second = _weightSecondMoment[i] = secondMomentDecay * _weightSecondMoment[i] + (1 - secondMomentDecay) * Color.Pow(_gradients[i].WeightGradient, 2);
                _weight[i] -= learningRate * first / (Color.Pow(second, 0.5f) + Utility.AsymptoteErrorColor);

                first = _biasFirstMoment[i] = firstMomentDecay * _biasFirstMoment[i] + (1 - firstMomentDecay) * _gradients[i].BiasGradient;
                second = _biasSecondMoment[i] = secondMomentDecay * _biasSecondMoment[i] + (1 - secondMomentDecay) * Color.Pow(_gradients[i].BiasGradient, 2);
                _bias[i] -= learningRate * first / (Color.Pow(second, 0.5f) + Utility.AsymptoteErrorColor);
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            //LoadCommon(false);
            MeanCPU();
            VarianceCPU();
            NormalizeCPU();
            //DisposeCommon();
        }

        /// <summary>
        /// Calculates the mean of each dimension using an <see cref="ILGPU"/> kernal.
        /// </summary>
        public void MeanGPU()
        {
            Accelerator accelerator = Utility.Accelerator;
            var sumKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<StaticLayerInfo>>(MeanKernal);


            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceSums[i] = accelerator.Allocate1D<float>(3);
                _deviceSums[i].MemSetToZero();

                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

                for (int j = 0; j < _batchSize; j++)
                {
                    sumKernal(index, _deviceInputs[i, j].View, _deviceSums[i].View, _deviceInfos[i].View);
                }
            }

            accelerator.Synchronize();
            for (int i = 0; i < _inputDimensions; i++)
            {
                _mean[i] = (Color)_deviceSums[i] / (Infos(i).Area * _batchSize);
                _deviceSums[i].Dispose();
            }
        }

        /// <summary>
        /// Calculates the mean of each dimension using the CPU.
        /// </summary>
        public void MeanCPU()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                _mean[i] = new Color(0);
                for (int j = 0; j < _batchSize; j++)
                {
                    _mean[i] += _inputs[i, j].Average();
                }
                _mean[i] /= _batchSize;
            }
        }

        /// <summary>
        /// Calculates the standard deviation of each dimension using an <see cref="ILGPU"/> kernal.
        /// </summary>
        public void VarianceGPU()
        {
            Accelerator accelerator = Utility.Accelerator;
            var varianceKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<StaticLayerInfo>>(VarianceKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceMeans[i] = accelerator.Allocate1D(new Color[] { _mean[i] });
                _deviceVariances[i] = accelerator.Allocate1D<float>(3);
                    _deviceVariances[i].MemSetToZero();

                    Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

                for (int j = 0; j < _batchSize; j++)
                {
                    varianceKernal(index, _deviceInputs[i, j].View, _deviceMeans[i].View, _deviceVariances[i].View, _deviceInfos[i].View);
                }
            }

            accelerator.Synchronize();
            for (int i = 0; i < _inputDimensions; i++)
            {
                _sigma[i] = Color.Pow((Color)_deviceVariances[i] / (Infos(i).Area * _batchSize) + Utility.AsymptoteErrorColor, 0.5f);
                _deviceMeans[i].Dispose();
                _deviceVariances[i].Dispose();
            }
        }

        /// <summary>
        /// Calculates the standard deviation of each dimension using the CPU.
        /// </summary>
        public void VarianceCPU()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                _sigma[i] = new Color(0);
                for (int j = 0; j < _batchSize; j++)
                {
                    for (int y = 0; y < _inputs[i, j].Length; y++)
                    {
                        for (int x = 0; x < _inputs[i, j].Width; x++)
                        {
                            _sigma[i] += Color.Pow(_inputs[i, j][x, y] - _mean[i], 2);
                        }
                    }
                }
                _sigma[i] = Color.Pow(_sigma[i] / (Infos(i).Area * _batchSize) + Utility.AsymptoteErrorColor, 0.5f);
            }
        }

        /// <summary>
        /// Normalizes the input <see cref="FeatureMap"/>s using an <see cref="ILGPU"/> kernal.
        /// </summary>
        public void NormalizeGPU()
        {
            Accelerator accelerator = Utility.Accelerator;
            var normalizeKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<StaticLayerInfo>>(ForwardKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index2D index = new(Infos(i).Width, Infos(i).Length);
                _deviceValues[i] = accelerator.Allocate1D(new Color[] { _mean[i], _weight[i] / _sigma[i], _bias[i] });

                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceOutputs[i, j] = _outputs[i, j].AllocateEmpty(accelerator);

                    normalizeKernal(index, _deviceInputs[i, j].View, _deviceOutputs[i, j].View, _deviceValues[i].View, _deviceInfos[i].View);
                }
            }

            accelerator.Synchronize();
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _outputs[i, j].CopyFromBuffer(_deviceOutputs[i, j]);
                    _deviceOutputs[i, j].Dispose();
                }
                _deviceValues[i].Dispose();
            }
        }

        /// <summary>
        /// Normalizes the input <see cref="FeatureMap"/>s on the CPU.
        /// </summary>
        public void NormalizeCPU()
        {
            for(int i = 0; i < _inputDimensions; i++)
            {
                for(int j = 0; j < _batchSize; j++)
                {
                    for(int y = 0; y < _inputs[i,j].Length; y++)
                    {
                        for(int x = 0; x < _inputs[i,j].Width; x++)
                        {
                            _outputs[i, j][x,y] = (_inputs[i, j][x,y] - _mean[i]) * _weight[i] / _sigma[i] + _bias[i];
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Loads the common device data onto the <see cref="Accelerator"/>.
        /// </summary>
        public void LoadCommon(bool inGradients)
        {
            Accelerator accelerator = Utility.Accelerator;

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = accelerator.Allocate1D(new StaticLayerInfo[] { Infos(i) });

                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInputs[i, j] = _inputs[i, j].Allocate(accelerator);
                    if(inGradients)
                    {
                        _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);
                    }
                }
            }
        }

        /// <summary>
        /// Disposes the common device data from the <see cref="Accelerator"/>.
        /// </summary>
        public void DisposeCommon()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInputs[i, j].Dispose();
                    if (_deviceInGradients[i, j] != null && !_deviceInGradients[i,j].IsDisposed)
                        _deviceInGradients[i, j].Dispose();
                }
                _deviceInfos[i].Dispose();
            }
        }

        /// <summary>
        /// Called when the layer is deserialized.
        /// Temporary function to allow for loading models that were created before Adam optimization was used implemented.
        /// </summary>
        /// <param name="context">The streaming context for deserialization.</param>
        [OnDeserialized]
        public void OnDeserialized(StreamingContext context)
        {
            if (_weightFirstMoment == null || _weightSecondMoment == null || _biasFirstMoment == null || _biasSecondMoment == null)
            {
                _weightFirstMoment = new ColorVector(_weight.Length);
                _weightSecondMoment = new ColorVector(_weight.Length);
                _biasFirstMoment = new ColorVector(_bias.Length);
                _biasSecondMoment = new ColorVector(_bias.Length);
            }
        }

        /// <inheritdoc/>
        public override void Reset()
        {
            _weightFirstMoment = new ColorVector(_inputDimensions);
            _weightSecondMoment = new ColorVector(_inputDimensions);
            _bias = new ColorVector(_inputDimensions);
            _biasFirstMoment = new ColorVector(_inputDimensions);
            _biasSecondMoment = new ColorVector(_inputDimensions);
            for (int i = 0; i < _inputDimensions; i++)
            {
                _weight[i] = new Color(1);
            }
        }

        /// <inheritdoc/>
        public override (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] inputs, FeatureMap[,] outGradients)
        {
            BaseStartup(inputs, outGradients);

            if (_weight == null)
            {
                _weight = new ColorVector(_inputDimensions);
                _weightFirstMoment = new ColorVector(_inputDimensions);
                _weightSecondMoment = new ColorVector(_inputDimensions);
                _bias = new ColorVector(_inputDimensions);
                _biasFirstMoment = new ColorVector(_inputDimensions);
                _biasSecondMoment = new ColorVector(_inputDimensions);
                for (int i = 0; i < _inputDimensions; i++)
                {
                    _weight[i] = new Color(1);
                }
            }

            _mean = new ColorVector(_inputDimensions);
            _sigma = new ColorVector(_inputDimensions);
            _gradients = new Gradients[_inputDimensions];

            _deviceInfos = new MemoryBuffer1D<StaticLayerInfo, Stride1D.Dense>[_inputDimensions];
            _deviceMeans = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions];
            _deviceGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions];
            _deviceValues = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions];
            _deviceSums = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions];
            _deviceVariances = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions];

            return (_outputs, _inGradients);
        }

        /// <summary>
        /// An ILGPU kernal to calculate the gradients for backpropagating the previous layer.
        /// </summary>
        /// <param name="index">The index of the current kernal calculation to be made.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="inGradient">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the incoming
        /// gradient from the following <see cref="Layer"/>.</param>
        /// <param name="outGradient">An <see cref="ArrayView1D{T, TStride}"/> of floats to sum the outgoing gradient.
        /// Because <see cref="Color"/> cannot be summed atomically, every three floats represents a single
        /// <see cref="Color"/> in the gradient.</param>
        /// <param name="values">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s used in the equation
        /// to calculate the outGradient.</param>
        /// <param name="info">The <see cref="StaticLayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void BackwardsKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> inGradient, ArrayView<float> outGradient, ArrayView<Color> values, ArrayView<StaticLayerInfo> info)
        {
            int mapsIndex = info[0].Index(index.X, index.Y);
            outGradient[mapsIndex * 3 + index.Z] = (inGradient[mapsIndex] * values[0] + values[1] * (input[mapsIndex] - values[2]) + values[3]).Clamp(1)[index.Z];
        }

        /// <summary>
        /// An ILGPU kernal for normalizing a <see cref="FeatureMap"/>.
        /// </summary>
        /// <param name="index">The index of the current kernal calculation to be made.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="normalized">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s to set for the outgoing
        /// convoluted <see cref="FeatureMap"/>.</param>
        /// <param name="values">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s used in the equation to
        /// calculate the normalized <see cref="Color"/>.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> normalized, ArrayView<Color> values, ArrayView<StaticLayerInfo> info)
        {
            int mapsIndex = info[0].Index(index.X, index.Y);
            normalized[mapsIndex] = (input[mapsIndex] - values[0]) * values[1] + values[2];
        }

        private static void GradientsKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> inGradient, ArrayView<Color> normalized, ArrayView<Color> mean, ArrayView<float> gradients, ArrayView<StaticLayerInfo> layer)
        {
            int gradientIndex = layer[0].Index(index.X, index.Y);
            float gradient = inGradient[gradientIndex][index.Z];
            Atomic.Add(ref gradients[index.Z], gradient * normalized[gradientIndex][index.Z]);
            Atomic.Add(ref gradients[index.Z + 3], gradient);
            Atomic.Add(ref gradients[index.Z + 6], gradient * (input[gradientIndex][index.Z] - mean[0][index.Z]));
        }

        private static void MeanKernal(Index3D index, ArrayView<Color> input, ArrayView<float> mean, ArrayView<StaticLayerInfo> info)
        {
            int inputIndex = info[0].Index(index.X, index.Y);
            Atomic.Add(ref mean[index.Z], input[inputIndex][index.Z]);
        }

        private static void VarianceKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> mean, ArrayView<float> variance, ArrayView<StaticLayerInfo> info)
        {
            int inputIndex = info[0].Index(index.X, index.Y);
            float difference = input[inputIndex][index.Z] - mean[0][index.Z];
            Atomic.Add(ref variance[index.Z], difference * difference);
        }

        /// <summary>
        /// Gets the <see cref="StaticLayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="StaticLayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="StaticLayerInfo"/> corresponding to an input dimension.</returns>
        private StaticLayerInfo Infos(int index)
        {
            return (StaticLayerInfo)_layerInfos[index];
        }

        /// <summary>
        /// The <see cref="Gradients"/> struct is a collection of different gradients used for backpropagating through <see cref="BatchNormalization"/> layers.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public class Gradients
        {
            private Color _weightGradient;
            private Color _biasGradient;
            private Color _sigmaGradient;

            /// <value>The gradient for the dimensions weight.</value>
            public Color WeightGradient { get => _weightGradient; set => _weightGradient = value; }

            /// <value>The gradient for the dimensions bias.</value>
            public Color BiasGradient { get => _biasGradient; set => _biasGradient = value; }

            /// <value>The gradient for the dimensions standard deviation.</value>
            public Color SigmaGradient { get => _sigmaGradient; set => _sigmaGradient = value; }

            /// <value>The gradient for the dimensions mean.</value>
            public Color MeanGradient { get; set; }

            /// <summary>
            /// Copies the pixel data from a <see cref="MemoryBuffer1D{T, TStride}"/> of floats.
            /// Because <see cref="Color"/> cannot be summed atomically on an <see cref="ILGPU"/> kernal, every three floats represents a single
            /// <see cref="Color"/> in the gradient. The <see cref="Gradients"/> is then treated as a <see cref="Span{T}"/> of floats, instead of
            /// a struct of <see cref="Color"/>.
            /// </summary>
            /// <param name="buffer">The <see cref="MemoryBuffer1D{T, TStride}"/> with the source floats.</param>
            public void CopyFromBuffer(MemoryBuffer1D<float, Stride1D.Dense> buffer)
            {
                unsafe
                {
                    fixed (void* startAddress = &_weightGradient)
                        buffer.AsArrayView<float>(0, 9).CopyToCPU(new Span<float>(startAddress, 9));
                }
            }
        }
    }
}