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
        private static readonly Action<Index1D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>> s_backwardsAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>>(BackwardsKernal);
        private static readonly Action<Index2D, ArrayView<Color>, ArrayView<float>, ArrayView<Color>, ArrayView<float>> s_gradientAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<float>, ArrayView<Color>, ArrayView<float>>(GradientsKernal);
        private static readonly Action<Index1D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>> s_normalizeAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>>(ForwardKernal);
        private static readonly Action<Index2D, ArrayView<Color>, ArrayView<float>> s_sumAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<float>>(MeanKernal);
        private static readonly Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>> s_varianceAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>>(VarianceKernal);
        
        private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceGradients;
        private MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceInputs;
        private MemoryBuffer1D<Color, Stride1D.Dense>[] _deviceMeans;
        private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceSums;
        private MemoryBuffer1D<Color, Stride1D.Dense>[] _deviceValues;
        private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceVariances;
        private Gradients[] _gradients;
        private FeatureMap[,] _inputs;
        private ColorVector _mean;
        private ColorVector _sigma;
        [JsonProperty] private Filter _weights;
        [JsonProperty] private Filter _bias;


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
            for(int i = 0; i < _inputDimensions; i++)
            {
                for(int j = 0; j < _batchSize; j++)
                    _deviceInputs[i, j] = _inputs[i, j].Allocate(Utility.Accelerator);
            }
            GPUGradients();
            GPUBackPropogate();
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                    _deviceInputs[i, j].Dispose();
            }
            UpdateWeights(learningRate,firstMomentDecay, secondMomentDecay);
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
            MeanGPU();
            VarianceGPU();
            NormalizeGPU();
        }

        public void GPUBackPropogate()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                float invM = 1f / (Infos(i).Area * _batchSize);

                _deviceValues[i] = Utility.Accelerator.Allocate1D(new Color[] { _weights[i] / _sigma[i], 2 * invM * _gradients[i].SigmaGradient, _mean[i], invM * _gradients[i].MeanGradient });

                Index1D index = new(Infos(i).Area);

                for (int j = 0; j < _batchSize; j++)
                {
                    s_backwardsAction(index, _deviceInputs[i, j].View, _buffers.InGradientsColor[i, j], _buffers.OutGradientsColor[i, j], _deviceValues[i].View);
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

                _deviceValues[i] = Utility.Accelerator.Allocate1D(new Color[] { _mean[i], _weights[i] / _sigma[i], _bias[i] });

                _deviceGradients[i] = Utility.Accelerator.Allocate1D<float>(9);
                _deviceGradients[i].MemSetToZero();
                Index2D index = new(Infos(i).Area, 3);

                for (int j = 0; j < _batchSize; j++)
                {
                    s_gradientAction(index, _deviceInputs[i, j].View, _buffers.InGradientsFloat[i, j], _deviceValues[i].View, _deviceGradients[i].View);
                }
            }

            Utility.Accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceValues[i].Dispose();
                _gradients[i] = new();
                _gradients[i].CopyFromBuffer(_deviceGradients[i]);
                _deviceGradients[i].Dispose();

                _gradients[i].SigmaGradient *= Color.Pow(_sigma[i], -1.5f) * _weights[i] * -0.5f;
                _gradients[i].MeanGradient = -_gradients[i].BiasGradient * _weights[i] / _sigma[i];
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

                Index2D index = new(Infos(i).Area, 3);

                for (int j = 0; j < _batchSize; j++)
                {
                    s_sumAction(index, _buffers.InputsColor[i, j], _deviceSums[i].View);
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
                Index1D index = new(Infos(i).Area);
                _deviceValues[i] = Utility.Accelerator.Allocate1D(new Color[] { _mean[i], _weights[i] / _sigma[i], _bias[i] });

                for (int j = 0; j < _batchSize; j++)
                {
                    s_normalizeAction(index, _buffers.InputsColor[i, j], _buffers.OutputsColor[i, j], _deviceValues[i].View);
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
        }

        /// <inheritdoc/>
        public override void Reset()
        {
            _weights.Reset(Color.One);
            _bias.Reset(Color.Zero);
        }

        /// <inheritdoc/>
        public override FeatureMap[,] Startup(FeatureMap[,] inputs, IOBuffers buffers)
        {
            BaseStartup(inputs, buffers);

            if (_weights == null)
            {
                _weights = new Filter(_inputDimensions, Color.One);
                _bias = new Filter(_inputDimensions, Color.Zero);
            }

            _mean = new ColorVector(_inputDimensions);
            _sigma = new ColorVector(_inputDimensions);
            _gradients = new Gradients[_inputDimensions];


            _inputs = inputs;
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
                _weights.SetGradient(i, _gradients[i].WeightGradient, learningRate, firstMomentDecay, secondMomentDecay);
                _bias.SetGradient(i, _gradients[i].BiasGradient, learningRate, firstMomentDecay, secondMomentDecay);
            }
        }

        public void FilterTest()
        {
            FeatureMap input = FilterTestSetup();

            _weights.TestFilterGradient(this, input, _outputs[0, 0], _buffers);
            _bias.TestFilterGradient(this, input, _outputs[0, 0], _buffers);
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

                Index2D index = new(Infos(i).Area, 3);

                for (int j = 0; j < _batchSize; j++)
                {
                    s_varianceAction(index, _buffers.InputsColor[i, j], _deviceMeans[i].View, _deviceVariances[i].View);
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
        private static void BackwardsKernal(Index1D index, ArrayView<Color> input, ArrayView<Color> inGradient, ArrayView<Color> outGradient, ArrayView<Color> values)
        {
            outGradient[index.X] = (inGradient[index.X] * values[0] + values[1] * (input[index.X] - values[2]) + values[3]).Clip(1);
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
        private static void ForwardKernal(Index1D index, ArrayView<Color> input, ArrayView<Color> normalized, ArrayView<Color> values)
        {
            normalized[index.X] = (input[index.X] - values[0]) * values[1] + values[2];
        }

        private static void GradientsKernal(Index2D index, ArrayView<Color> input, ArrayView<float> inGradient, ArrayView<Color> values, ArrayView<float> gradients)
        {
            float meanOffset = input[index.X][index.Y] - values[0][index.Y];
            float gradient = inGradient[3 * index.X + index.Y];
            
            float normalized = meanOffset * values[1][index.Y] + values[2][index.Y];

            Atomic.Add(ref gradients[index.Y], gradient * normalized);
            Atomic.Add(ref gradients[index.Y + 3], gradient);
            Atomic.Add(ref gradients[index.Y + 6], gradient * meanOffset);
        }

        private static void MeanKernal(Index2D index, ArrayView<Color> input, ArrayView<float> mean)
        {
            Atomic.Add(ref mean[index.Y], input[index.X][index.Y]);
        }

        private static void VarianceKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> mean, ArrayView<float> variance)
        {
            float difference = input[index.X][index.Y] - mean[0][index.Y];
            Atomic.Add(ref variance[index.Y], difference * difference);
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