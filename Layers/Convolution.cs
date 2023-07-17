using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="Convolution"/> class is a <see cref="Layer"/> that performs the titular convolutions of a convolutional
    /// neural network, by passing <see cref="FeatureMap"/>s through a variety of filters.
    /// </summary>
    [Serializable]
    public class Convolution : Layer, IPrimaryLayer
    {
        private MemoryBuffer1D<LayerInfo, Stride1D.Dense>[] _deviceInfos;
        private FeatureMap[,] _inputs;
        private readonly int _dimensionsMultiplier;
        [JsonProperty] private Weights[] _filters;

        /// <summary>
        /// Initializes a new instance of the <see cref="Convolution"/> layer.
        /// </summary>
        /// <param name="filterSize">The width and height of a filter.</param>
        /// <param name="stride">The amount of movement over the image for each filter pass.</param>
        /// <param name="outputDimensionsMultiplier">A factor relating the number of input layers to the number of output layers.
        /// Must be positive. To reduce the number of output dimensions, use a <see cref="Summation"/> layer afterwards.</param>
        public Convolution(int filterSize, int stride, int outputDimensionsMultiplier) : base(filterSize, stride)
        {
            if (outputDimensionsMultiplier < 1)
            {
                throw new ArgumentException("Dimension multiplier must be greater than or equal to 1.");
            }
            _dimensionsMultiplier = outputDimensionsMultiplier;
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor]
        private Convolution() : base()
        {
        }

        public static Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>> BackwardsFilterAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsFilterKernel);

        public static Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>> BackwardsOutGradientAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsGradientKernel);

        public static Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>> ForwardAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>>(ForwardKernel);

        /// <inheritdoc/>
        public override string Name => "Convolutional Layer";

        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            if (learningRate <= 0)
                BackwardsNoUpdate();
            else
                BackwardsUpdate(learningRate, firstMomentDecay, secondMomentDecay);
        }

        /// <summary>
        /// Backpropagates by updating the filters of the <see cref="Convolution"/> layer, but without calculating the gradient for further
        /// propagation. Used in the special case of the first layer of a <see cref="Network"/> to save time.
        /// </summary>
        /// <param name="learningRate">Controls how much the layer is updated with each backpropagation.</param>
        public void BackwardsUpdateOnly(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            for (int i = 0; i < _outputDimensions; i++)
            {
                Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
                for (int j = 0; j < _batchSize; j++)
                {
                    BackwardsFilterAction(index, _buffers.InGradientsFloat[i, j], _inputs[i % _inputDimensions, j].GetArrayView<float>(), _filters[i].GradientGPU<float>(), _deviceInfos[i % _inputDimensions].View);
                }
            }

            Synchronize();
            DecrementCacheabble(_inputs, (uint)(_outputDimensions / _inputDimensions));

            for (int i = 0; i < _outputDimensions; i++)
            {
                _filters[i].DisposeGradient(_batchSize);
                _filters[i].UpdateWeights(learningRate, firstMomentDecay, secondMomentDecay);
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index1D index = new(Infos(i).InputArea);
                for (int j = 0; j < _batchSize; j++)
                {
                    GPUManager.CopyAction(index, _buffers.InputsFloat[i, j], _inputs[i, j].GetArrayViewEmpty<float>());
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
                for (int j = 0; j < _batchSize; j++)
                {
                    ForwardAction(index, _buffers.InputsFloat[i % _inputDimensions, j], _buffers.OutputsFloat[i, j], _filters[i].WeightsGPU<float>(), _deviceInfos[i % _inputDimensions].View);
                }
            }
            Synchronize();
            DecrementCacheabble(_inputs);

            for(int i = 0; i < _outputDimensions; i++)
            {
                _filters[i].DisposeWeights(_batchSize);
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
            float variance = 2f / (_filterSize * _filterSize);
            float stdDev = MathF.Sqrt(variance);

            for (int i = 0; i < _outputDimensions; i++)
            {
                _filters[i].Reset(0, stdDev);
            }
        }

        /// <inheritdoc/>
        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, uint batchSize)
        {
            if (_filters == null)
            {
                BaseStartup(inputShapes, buffers, batchSize, _dimensionsMultiplier);
                _filters = new Weights[_outputDimensions];

                float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize + _inputDimensions * _filterSize * _filterSize);
                float stdDev = MathF.Sqrt(variance);

                int filterArea = _filterSize * _filterSize;

                for (int i = 0; i < _filters.Length; i++)
                {
                    _filters[i] = new Weights(filterArea, 0, 0.02f);
                }
            }
            else
            {
                BaseStartup(inputShapes, buffers, batchSize, _filters.Length / inputShapes.Length);
            }

            _deviceInfos = new MemoryBuffer1D<LayerInfo, Stride1D.Dense>[_inputDimensions];
            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = GPUManager.Accelerator.Allocate1D(new LayerInfo[] { Infos(i) });
            }

            _inputs = new FeatureMap[_inputDimensions, batchSize];
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < batchSize; j++)
                {
                    _inputs[i, j] = new FeatureMap(inputShapes[i]);
                }
            }

            return _outputShapes;
        }

        /// <summary>
        /// Gets the <see cref="LayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="LayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="LayerInfo"/> corresponding to an input dimension.</returns>
        private LayerInfo Infos(int index)
        {
            return (LayerInfo)_layerInfos[index % _inputDimensions];
        }

        /// <summary>
        /// An ILGPU kernel to update the <see cref="Convolution"/>'s filters.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
        /// <param name="inGradient">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the incoming
        /// gradient from the following <see cref="Layer"/>.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="filterGradient">An <see cref="ArrayView1D{T, TStride}"/> of floats to sum the gradient of the filters.
        /// Because <see cref="Color"/> cannot be summed atomically, every three floats represents a single
        /// <see cref="Color"/> in the gradient.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void BackwardsFilterKernel(Index2D index, ArrayView<float> inGradient, ArrayView<float> input, ArrayView<float> filterGradient, ArrayView<LayerInfo> info)
        {
            float dL = inGradient[info[0].OutputIndex(index.X, index.Y)] * info[0].InverseKSquared;

            for (int j = 0; j < info[0].FilterSize; j++)
            {
                for (int i = 0; i < info[0].FilterSize; i++)
                {
                    if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                    {
                        int filterIndex = info[0].FilterIndex(i, j);
                        float dK = dL * input[inputIndex];
                        Atomic.Add(ref filterGradient[filterIndex], dK);
                    }
                }
            }
        }

        /// <summary>
        /// An ILGPU kernel to calculate the gradients for backpropagating the previous layer.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
        /// <param name="inGradient">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the incoming
        /// gradient from the following <see cref="Layer"/>.</param>
        /// <param name="filter">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing one of the
        /// <see cref="Convolution"/>'s filters.</param>
        /// <param name="outGradient">An <see cref="ArrayView1D{T, TStride}"/> of floats to sum the outgoing gradient.
        /// Because <see cref="Color"/> cannot be summed atomically, every three floats represents a single
        /// <see cref="Color"/> in the gradient.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void BackwardsGradientKernel(Index2D index, ArrayView<float> inGradient, ArrayView<float> filter, ArrayView<float> outGradient, ArrayView<LayerInfo> info)
        {
            float dL = inGradient[info[0].OutputIndex(index.X, index.Y)] * info[0].InverseKSquared;

            for (int j = 0; j < info[0].FilterSize; j++)
            {
                for (int i = 0; i < info[0].FilterSize; i++)
                {
                    if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                    {
                        int filterIndex = info[0].FilterIndex(i, j);
                        float dP = dL * filter[filterIndex];
                        Atomic.Add(ref outGradient[inputIndex], dP);
                    }
                }
            }
        }

        /// <summary>
        /// An ILGPU kernel for convoluting a <see cref="FeatureMap"/>.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="convoluted">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s to set for the outgoing
        /// convoluted <see cref="FeatureMap"/>.</param>
        /// <param name="filter">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing one of the
        /// <see cref="Convolution"/>'s filters.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void ForwardKernel(Index2D index, ArrayView<float> input, ArrayView<float> convoluted, ArrayView<float> filter, ArrayView<LayerInfo> info)
        {
            float sum = 0;

            for (int j = 0; j < info[0].FilterSize; j++)
            {
                for (int i = 0; i < info[0].FilterSize; i++)
                {
                    if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                        sum += filter[info[0].FilterIndex(i, j)] * input[inputIndex];
                }
            }

            convoluted[info[0].OutputIndex(index.X, index.Y)] = sum * info[0].InverseKSquared;
        }
        /// <summary>
        /// Backpropagates through the layer without updating any of the filter weights. Called when learning rate is zero.
        /// </summary>
        private void BackwardsNoUpdate()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                for(int j = 0; j < _batchSize; j++)
                {
                    _buffers.OutGradientsFloat[i, j].SubView(0, Infos(i).InputArea).MemSetToZero();
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
                for (int j = 0; j < _batchSize; j++)
                {
                    BackwardsOutGradientAction(index, _buffers.InGradientsFloat[i, j], _filters[i].WeightsGPU<float>(), _buffers.OutGradientsFloat[i % _inputDimensions, j], _deviceInfos[i % _inputDimensions].View);
                }
            }
            Synchronize();

            for (int i = 0; i < _outputDimensions; i++)
            {
                _filters[i].DisposeWeights(_batchSize);
            }
        }

        /// <summary>
        /// Perform standard backpropagation through the layer, updating it's weights. Called when learning rate is greater than 0.
        /// </summary>
        /// <param name="learningRate">The overall learning rate for the layer updates, corrected for the influence of bias in the first and second moments.</param>
        /// <param name="firstMomentDecay">The exponential decay rate for the first moment.</param>
        /// <param name="secondMomentDecay">The exponential decay rate for the second moment.</param>
        private void BackwardsUpdate(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _buffers.OutGradientsFloat[i, j].SubView(0, Infos(i).InputArea).MemSetToZero();
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
                for (int j = 0; j < _batchSize; j++)
                {
                    BackwardsOutGradientAction(index, _buffers.InGradientsFloat[i, j], _filters[i].WeightsGPU<float>(), _buffers.OutGradientsFloat[i % _inputDimensions, j], _deviceInfos[i % _inputDimensions].View);
                    BackwardsFilterAction(index, _buffers.InGradientsFloat[i, j], _inputs[i % _inputDimensions, j].GetArrayView<float>(), _filters[i].GradientGPU<float>(), _deviceInfos[i % _inputDimensions].View);
                }
            }

            Synchronize();
            DecrementCacheabble(_inputs, (uint)(_outputDimensions / _inputDimensions));

            for (int i = 0; i < _outputDimensions; i++)
            {
                _filters[i].DisposeGradient(_batchSize);
                _filters[i].DisposeWeights(_batchSize);
                _filters[i].UpdateWeights(learningRate, firstMomentDecay, secondMomentDecay);
            }
        }

        public void FilterTest(int outputMultiplier)
        {
            FeatureMap[,] input = FilterTestSetup(outputMultiplier);

            for (int i = 0; i < outputMultiplier; i++)
            {
                FeatureMap output = new(_outputShapes[i]);

                _filters[i].TestFilterGradient(this, input, output, i, _buffers);
            }
        }
    }
}