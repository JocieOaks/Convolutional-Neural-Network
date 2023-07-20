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
        /*private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>> s_backwardsAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>(BackwardsKernel);
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>> s_gradientAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>(GradientsKernel);
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> s_normalizeAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(ForwardKernel);
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> s_sumAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(MeanKernel);
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> s_varianceAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VarianceKernel);

        [JsonProperty] private Weights _bias;
        private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceGradients;
        private MemoryBuffer1D<float, Stride1D.Dense>[] _deviceValues;
        private Gradients[] _gradients;
        private FeatureMap[,] _inputs;
        private Vector _mean;
        private Vector _sigma;
        [JsonProperty] private Weights _weights;
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
            GPUGradients();
            GPUBackPropogate();
            UpdateWeights(learningRate,firstMomentDecay, secondMomentDecay);
        }

        public void FilterTest()
        {
            FeatureMap[,] input = FilterTestSetup(1);

            FeatureMap output = new FeatureMap(_outputShapes[0]);

            _weights.TestFilterGradient(this, input, output, 0, _buffers);
            _bias.TestFilterGradient(this, input, output, 0, _buffers);
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index1D index = new(Infos(i).Area);
                for (int j = 0; j < _batchSize; j++)
                {
                    GPU.GPUManager.CopyAction(index, _buffers.Input[i, j], _inputs[i, j].GetArrayViewEmpty<float>());
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

                _deviceValues[i] = GPU.GPUManager.Accelerator.Allocate1D(new float[] { _weights[i] / _sigma[i], 2 * invM * _gradients[i].SigmaGradient, _mean[i], invM * _gradients[i].MeanGradient });

                Index1D index = new(Infos(i).Area);

                for (int j = 0; j < _batchSize; j++)
                {
                    s_backwardsAction(index, _inputs[i, j].GetArrayView<float>(), _buffers.InGradient[i, j], _buffers.OutGradient[i, j], _deviceValues[i].View);
                }
            }
            Synchronize();
            DecrementCacheabble(_inputs);

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

                _deviceValues[i] = GPU.GPUManager.Accelerator.Allocate1D(new float[] { _mean[i], _weights[i] / _sigma[i], _bias[i] });

                _deviceGradients[i] = GPU.GPUManager.Accelerator.Allocate1D<float>(9);
                _deviceGradients[i].MemSetToZero();
                Index1D index = new(Infos(i).Area);

                for (int j = 0; j < _batchSize; j++)
                {
                    s_gradientAction(index, _inputs[i, j].GetArrayView<float>(), _buffers.InGradient[i, j], _deviceValues[i].View, _deviceGradients[i].View);
                }
            }
            Synchronize();
            DecrementCacheabble(_inputs);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceValues[i].Dispose();
                _gradients[i] = new();
                _gradients[i].CopyFromBuffer(_deviceGradients[i]);
                _deviceGradients[i].Dispose();

                _gradients[i].SigmaGradient *= MathF.Pow(_sigma[i], -1.5f) * _weights[i] * -0.5f;
                _gradients[i].MeanGradient = -_gradients[i].BiasGradient * _weights[i] / _sigma[i];
            }
        }

        /// <summary>
        /// Calculates the mean of each dimension using an <see cref="ILGPU"/> kernel.
        /// </summary>
        public void MeanGPU()
        {
            var buffer = _mean.GetArrayView<float>();
            buffer.MemSetToZero();
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index1D index = new(Infos(i).Area);

                for (int j = 0; j < _batchSize; j++)
                {
                    s_sumAction(index, _buffers.Input[i, j], buffer.SubView(i));
                }
            }
            Synchronize();
            _mean.SyncCPU(buffer);
            for (int i = 0; i < _inputDimensions; i++)
            {
                _mean[i] = _mean[i] / (Infos(i).Area * (int)_batchSize);
            }
            _mean.UpdateIfAllocated();
            _mean.DecrementLiveCount(1);
        }

        /// <summary>
        /// Normalizes the input <see cref="FeatureMap"/>s using an <see cref="ILGPU"/> kernel.
        /// </summary>
        public void NormalizeGPU()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index1D index = new(Infos(i).Area);
                _deviceValues[i] = GPU.GPUManager.Accelerator.Allocate1D(new float[] { _mean[i], _weights[i] / _sigma[i], _bias[i] });

                for (int j = 0; j < _batchSize; j++)
                {
                    s_normalizeAction(index, _buffers.Input[i, j], _buffers.Output[i, j], _deviceValues[i].View);
                }
            }
            Synchronize();
            DecrementCacheabble(_inputs);

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
            _weights.Reset(1);
            _bias.Reset(0);
        }

        /// <inheritdoc/>
        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, uint batchSize)
        {
            BaseStartup(inputShapes, buffers, batchSize);

            if (_weights == null)
            {
                _weights = new Weights(_inputDimensions, 1);
                _bias = new Weights(_inputDimensions, 0);
            }

            _mean = new Vector(_inputDimensions);
            _sigma = new Vector(_inputDimensions);
            _gradients = new Gradients[_inputDimensions];

            _deviceGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions];
            _deviceValues = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions];
            _inputs = new FeatureMap[_inputDimensions, batchSize];
            for(int i = 0; i < _inputDimensions; i++)
            {
                for(int j = 0; j < batchSize; j++)
                {
                    _inputs[i, j] = new FeatureMap(inputShapes[i]);
                }
            }

            return _outputShapes;
        }

        public void UpdateWeights(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                _weights.SetGradient(i, _gradients[i].WeightGradient, learningRate, firstMomentDecay, secondMomentDecay);
                _bias.SetGradient(i, _gradients[i].BiasGradient, learningRate, firstMomentDecay, secondMomentDecay);
            }
        }
        /// <summary>
        /// Calculates the standard deviation of each dimension using an <see cref="ILGPU"/> kernel.
        /// </summary>
        public void VarianceGPU()
        {
            var varBuffer = _sigma.GetArrayView<float>();
            varBuffer.MemSetToZero();
            var meanBuffer = _mean.GetArrayView<float>();
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index1D index = new(Infos(i).Area);

                for (int j = 0; j < _batchSize; j++)
                {
                    s_varianceAction(index, _buffers.Input[i, j], meanBuffer.SubView(i), varBuffer.SubView(i));
                }
            }
            Synchronize();
            _sigma.SyncCPU(varBuffer);
            for (int i = 0; i < _inputDimensions; i++)
            {
                _sigma[i] = MathF.Pow(_sigma[i] / (Infos(i).Area * (int)_batchSize) + Utility.ASYMPTOTEERRORCORRECTION, 0.5f);
            }
            _sigma.UpdateIfAllocated();
            _sigma.DecrementLiveCount();
            _mean.DecrementLiveCount();
        }

        /// <summary>
        /// An ILGPU kernel to calculate the gradients for backpropagating the previous layer.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
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
        private static void BackwardsKernel(Index3D index, ArrayView<float> input, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<float> values, ArrayView<StaticLayerInfo> infoView)
        {
            StaticLayerInfo info = infoView[index.X];
            int ind = index.X * index.Y * info.Area + index.Z;
            outGradient[ind] = inGradient[ind] * values[0] + values[1] * (input[ind] - values[2]) + values[3];
        }

        /// <summary>
        /// An ILGPU kernel for normalizing a <see cref="FeatureMap"/>.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="normalized">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s to set for the outgoing
        /// convoluted <see cref="FeatureMap"/>.</param>
        /// <param name="values">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s used in the equation to
        /// calculate the normalized <see cref="Color"/>.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void ForwardKernel(Index3D index, ArrayView<float> input, ArrayView<float> normalized, ArrayView<float> values, ArrayView<StaticLayerInfo> infoView)
        {
            StaticLayerInfo info = infoView[index.X];
            int ind = index.X * index.Y * info.Area + index.Z;
            normalized[ind] = (input[ind] - values[0]) * values[1] + values[2];
        }

        private static void GradientsKernel(Index3D index, ArrayView<float> input, ArrayView<float> inGradient, ArrayView<float> values, ArrayView<float> gradients, ArrayView<StaticLayerInfo> infoView)
        {
            StaticLayerInfo info = infoView[index.X];
            int ind = index.X * index.Y * info.Area + index.Z;

            float meanOffset = input[ind] - values[0];
            float gradient = inGradient[ind];
            
            float normalized = meanOffset * values[1] + values[2];

            Atomic.Add(ref gradients[0], gradient * normalized);
            Atomic.Add(ref gradients[1], gradient);
            Atomic.Add(ref gradients[2], gradient * meanOffset);
        }

        private static void MeanKernel(Index1D index, ArrayView<float> input, ArrayView<float> mean)
        {
            Atomic.Add(ref mean[0], input[index.X]);
        }

        private static void VarianceKernel(Index1D index, ArrayView<float> input, ArrayView<float> mean, ArrayView<float> variance)
        {
            float difference = input[index.X] - mean[0];
            Atomic.Add(ref variance[0], difference * difference);
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
            private float _weightGradient;
            private float _biasGradient;
            private float _sigmaGradient;

            /// <value>The gradient for the dimensions weight.</value>
            public float WeightGradient { get => _weightGradient; set => _weightGradient = value; }

            /// <value>The gradient for the dimensions bias.</value>
            public float BiasGradient { get => _biasGradient; set => _biasGradient = value; }

            /// <value>The gradient for the dimensions standard deviation.</value>
            public float SigmaGradient { get => _sigmaGradient; set => _sigmaGradient = value; }

            /// <value>The gradient for the dimensions mean.</value>
            public float MeanGradient { get; set; }

            /// <summary>
            /// Copies the pixel data from a <see cref="MemoryBuffer1D{T, TStride}"/> of floats.
            /// Because <see cref="Color"/> cannot be summed atomically on an <see cref="ILGPU"/> kernel, every three floats represents a single
            /// <see cref="Color"/> in the gradient. The <see cref="Gradients"/> is then treated as a <see cref="Span{T}"/> of floats, instead of
            /// a struct of <see cref="Color"/>.
            /// </summary>
            /// <param name="buffer">The <see cref="MemoryBuffer1D{T, TStride}"/> with the source floats.</param>
            public void CopyFromBuffer(MemoryBuffer1D<float, Stride1D.Dense> buffer)
            {
                unsafe
                {
                    fixed (void* startAddress = &_weightGradient)
                        buffer.AsArrayView<float>(0, 3).CopyToCPU(new Span<float>(startAddress, 3));
                }
            }
        }*/
        public override string Name => throw new NotImplementedException();

        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            throw new NotImplementedException();
        }

        public override void Forward()
        {
            throw new NotImplementedException();
        }

        public override void Reset()
        {
            throw new NotImplementedException();
        }

        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, int batchSize)
        {
            throw new NotImplementedException();
        }
    }
}