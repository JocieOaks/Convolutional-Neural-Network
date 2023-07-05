using ConvolutionalNeuralNetwork.DataTypes;
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
        private static readonly Action<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<float>, ArrayView<Color>, ArrayView<StaticLayerInfo>> s_backwardsAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<float>, ArrayView<Color>, ArrayView<StaticLayerInfo>>(BackwardsKernal);
        private static readonly Action<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<Color>, ArrayView<float>, ArrayView<StaticLayerInfo>> s_gradientAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<Color>, ArrayView<float>, ArrayView<StaticLayerInfo>>(GradientsKernal);
        private static readonly Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<StaticLayerInfo>> s_normalizeAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<StaticLayerInfo>>(ForwardKernal);
        private static readonly Action<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<StaticLayerInfo>> s_sumAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<StaticLayerInfo>>(MeanKernal);
        private static readonly Action<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<StaticLayerInfo>> s_varianceAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<StaticLayerInfo>>(VarianceKernal);
        
        [JsonProperty] private ColorVector _bias;
        [JsonProperty] private ColorVector _biasFirstMoment;
        [JsonProperty] private ColorVector _biasSecondMoment;
        private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceGradients;
        private MemoryBuffer1D<StaticLayerInfo, Stride1D.Dense>[] _deviceInfos;
        private MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceInputs;
        private MemoryBuffer1D<Color, Stride1D.Dense>[] _deviceMeans;
        private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceSums;
        private MemoryBuffer1D<Color, Stride1D.Dense>[] _deviceValues;
        private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceVariances;
        private Gradients[] _gradients;
        private FeatureMap[,] _inputs;
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
            for(int i = 0; i < _inputDimensions; i++)
            {
                for(int j = 0; j < _batchSize; j++)
                    _deviceInputs[i, j] = _inputs[i, j].Allocate(Utility.Accelerator);
            }
            GPUGradients();
            GPUBackPropogate();
            DisposeCommon();
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                    _deviceInputs[i, j].Dispose();
            }
            UpdateWeights(learningRate,firstMomentDecay, secondMomentDecay);
        }
        /// <summary>
        /// Disposes the common device data from the <see cref="Accelerator"/>.
        /// </summary>
        public void DisposeCommon()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _inputs[i, j].CopyFromBuffer(_buffers.InputsColor[i, j]);
                }
            }

            LoadCommon(false);
            MeanGPU();
            VarianceGPU();
            NormalizeGPU();
            DisposeCommon();
        }

        public void GPUBackPropogate()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                float invM = 1f / (Infos(i).Area * _batchSize);

                _deviceValues[i] = Utility.Accelerator.Allocate1D(new Color[] { _weight[i] / _sigma[i], 2 * invM * _gradients[i].SigmaGradient, _mean[i], invM * _gradients[i].MeanGradient });

                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

                for (int j = 0; j < _batchSize; j++)
                {
                    s_backwardsAction(index, _deviceInputs[i, j].View, _buffers.InGradientsFloat[i, j], _buffers.OutGradientsFloat[i, j], _deviceValues[i].View, _deviceInfos[i].View);
                }
            }

            Utility.Accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceValues[i].Dispose();
            }
        }

        public void GPUGradients()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                float invM = 1f / (Infos(i).Area * _batchSize);

                _deviceValues[i] = Utility.Accelerator.Allocate1D(new Color[] { _mean[i], _weight[i] / _sigma[i], _bias[i] });

                _deviceGradients[i] = Utility.Accelerator.Allocate1D<float>(9);
                _deviceGradients[i].MemSetToZero();
                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

                for (int j = 0; j < _batchSize; j++)
                {
                    s_gradientAction(index, _deviceInputs[i, j].View, _buffers.InGradientsFloat[i, j], _deviceValues[i].View, _deviceGradients[i].View, _deviceInfos[i].View);
                }
            }

            Utility.Accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceValues[i].Dispose();
                _gradients[i] = new();
                _gradients[i].CopyFromBuffer(_deviceGradients[i]);
                _deviceGradients[i].Dispose();

                _gradients[i].SigmaGradient *= Color.Pow(_sigma[i], -1.5f) * _weight[i] * -0.5f;
                _gradients[i].MeanGradient = -_gradients[i].BiasGradient * _weight[i] / _sigma[i];
            }
        }
        /// <summary>
        /// Loads the common device data onto the <see cref="Accelerator"/>.
        /// </summary>
        public void LoadCommon(bool inGradients)
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = Utility.Accelerator.Allocate1D(new StaticLayerInfo[] { Infos(i) });
            }
        }

        /// <summary>
        /// Calculates the mean of each dimension using an <see cref="ILGPU"/> kernal.
        /// </summary>
        public void MeanGPU()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceSums[i] = Utility.Accelerator.Allocate1D<float>(3);
                _deviceSums[i].MemSetToZero();

                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

                for (int j = 0; j < _batchSize; j++)
                {
                    s_sumAction(index, _buffers.InputsColor[i, j], _deviceSums[i].View, _deviceInfos[i].View);
                }
            }

            Utility.Accelerator.Synchronize();
            for (int i = 0; i < _inputDimensions; i++)
            {
                _mean[i] = (Color)_deviceSums[i] / (Infos(i).Area * _batchSize);
                _deviceSums[i].Dispose();
            }
        }

        /// <summary>
        /// Normalizes the input <see cref="FeatureMap"/>s using an <see cref="ILGPU"/> kernal.
        /// </summary>
        public void NormalizeGPU()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index2D index = new(Infos(i).Width, Infos(i).Length);
                _deviceValues[i] = Utility.Accelerator.Allocate1D(new Color[] { _mean[i], _weight[i] / _sigma[i], _bias[i] });

                for (int j = 0; j < _batchSize; j++)
                {
                    s_normalizeAction(index, _buffers.InputsColor[i, j], _buffers.OutputsColor[i, j], _deviceValues[i].View, _deviceInfos[i].View);
                }
            }

            Utility.Accelerator.Synchronize();
            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceValues[i].Dispose();
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
        public override FeatureMap[,] Startup(FeatureMap[,] inputs, IOBuffers buffers)
        {
            BaseStartup(inputs, buffers);

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


            _inputs = inputs;
            _deviceInfos = new MemoryBuffer1D<StaticLayerInfo, Stride1D.Dense>[_inputDimensions];
            _deviceMeans = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions];
            _deviceGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions];
            _deviceValues = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions];
            _deviceSums = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions];
            _deviceVariances = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions];
            _deviceInputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];

            return _outputs;
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

        /// <summary>
        /// Calculates the standard deviation of each dimension using an <see cref="ILGPU"/> kernal.
        /// </summary>
        public void VarianceGPU()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceMeans[i] = Utility.Accelerator.Allocate1D(new Color[] { _mean[i] });
                _deviceVariances[i] = Utility.Accelerator.Allocate1D<float>(3);
                    _deviceVariances[i].MemSetToZero();

                    Index3D index = new(Infos(i).Width, Infos(i).Length, 3);

                for (int j = 0; j < _batchSize; j++)
                {
                    s_varianceAction(index, _buffers.InputsColor[i, j], _deviceMeans[i].View, _deviceVariances[i].View, _deviceInfos[i].View);
                }
            }

            Utility.Accelerator.Synchronize();
            for (int i = 0; i < _inputDimensions; i++)
            {
                _sigma[i] = Color.Pow((Color)_deviceVariances[i] / (Infos(i).Area * _batchSize) + Utility.AsymptoteErrorColor, 0.5f);
                _deviceMeans[i].Dispose();
                _deviceVariances[i].Dispose();
            }
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
        private static void BackwardsKernal(Index3D index, ArrayView<Color> input, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<Color> values, ArrayView<StaticLayerInfo> info)
        {
            int mapsIndex = info[0].Index(index.X, index.Y);
            outGradient[3 * mapsIndex + index.Z] = (inGradient[3 * mapsIndex + index.Z] * values[0] + values[1] * (input[mapsIndex] - values[2]) + values[3]).Clamp(1)[index.Z];
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

        private static void GradientsKernal(Index3D index, ArrayView<Color> input, ArrayView<float> inGradient, ArrayView<Color> values, ArrayView<float> gradients, ArrayView<StaticLayerInfo> layer)
        {
            int inputIndex = layer[0].Index(index.X, index.Y);

            float meanOffset = input[inputIndex][index.Z] - values[0][index.Z];
            float gradient = inGradient[3 * inputIndex + index.Z];
            
            float normalized = meanOffset * values[1][index.Z] + values[2][index.Z];

            Atomic.Add(ref gradients[index.Z], gradient * normalized);
            Atomic.Add(ref gradients[index.Z + 3], gradient);
            Atomic.Add(ref gradients[index.Z + 6], gradient * meanOffset);
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