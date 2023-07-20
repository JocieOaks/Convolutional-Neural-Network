using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
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
        private readonly int _dimensionsMultiplier;
        private ArrayView<LayerInfo> _deviceInfos;
        [JsonProperty] private Weights _filters;
        private Vector _inputCopy;

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

        public static Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>> BackwardsFilterAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsFilterKernel);

        public static Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>> BackwardsOutGradientAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsGradientKernel);

        public static Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>> ForwardAction { get; } = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>>(ForwardKernel);

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
            Index3D index = new(_batchSize, _outputDimensions, Infos(0).OutputArea);

            BackwardsFilterAction(index, _buffers.InGradient, _inputCopy.GetArrayView<float>(), _filters.GradientGPU<float>(), _deviceInfos);

            Index3D biasIndex = new(_outputDimensions * _outputShapes[0].Area, 1, _batchSize);
            GPUManager.AddAction(biasIndex, _bias.GradientGPU<float>(), _buffers.InGradient, _outputDimensions * _outputShapes[0].Area);

            Synchronize();

            _inputCopy.DecrementLiveCount();
            _bias.DecrementLiveGradient();
            _bias.UpdateWeights(learningRate, firstMomentDecay, secondMomentDecay);

            _filters.DecrementLiveGradient();
            _filters.UpdateWeights(learningRate, firstMomentDecay, secondMomentDecay);
        }

        public void FilterTest(int outputMultiplier, int batchSize, int inputSize)
        {
            (Shape[] input, Shape[] output) = FilterTestSetup(outputMultiplier, batchSize, inputSize);

            _filters.TestFilterGradient(this, input, output, _buffers);
            _bias.TestFilterGradient(this, input, output, _buffers);
            
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            Index1D copyIndex = new(_inputDimensions * Infos(0).InputArea * _batchSize);
            GPUManager.CopyAction(copyIndex, _buffers.Input, _inputCopy.GetArrayViewEmpty<float>());

            Index3D index = new(_batchSize, _outputDimensions, Infos(0).OutputArea);
            ForwardAction(index, _buffers.Input, _buffers.Output, _filters.WeightsGPU<float>(), _deviceInfos);

            Index3D biasIndex = new(_outputDimensions * _outputShapes[0].Area, _batchSize, 1);
            GPUManager.AddAction(biasIndex, _buffers.Output, _bias.WeightsGPU<float>(), _outputDimensions * _outputShapes[0].Area);

            Synchronize();

            _bias.DecrementLiveWeights();

            _inputCopy.DecrementLiveCount();

            _filters.DecrementLiveWeights();
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
                _filters.Reset(0, stdDev);
            }
        }

        [JsonProperty] private Weights _bias;

        /// <inheritdoc/>
        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, int batchSize)
        {
            if (_filters == null)
            {
                BaseStartup(inputShapes, buffers, batchSize, _dimensionsMultiplier);

                float variance = 0.6666f / (_outputDimensions * _filterSize * _filterSize + _inputDimensions * _filterSize * _filterSize);
                float stdDev = MathF.Sqrt(variance);

                _filters = new Weights(_filterSize * _filterSize * _outputDimensions, 0, 0.05f);
            }
            else
            {
                BaseStartup(inputShapes, buffers, batchSize, _filters.Length / _filterSize / _filterSize / inputShapes.Length);
            }

            _bias ??= new Weights(_outputDimensions * _outputShapes[0].Area, 0);

            _deviceInfos = GPUManager.Accelerator.Allocate1D(Array.ConvertAll(_layerInfos, info => (LayerInfo)info)).View;

            _inputCopy = new Vector(_inputDimensions * _batchSize * inputShapes[0].Area);

            return _outputShapes;
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
        private static void BackwardsFilterKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> input, ArrayView<float> filterGradient, ArrayView<LayerInfo> infoView)
        {
            LayerInfo info = infoView[0];
            (int inputOffset, int inGradientOffset) = info.GetOffset(index.X, index.Y);

            float dL = inGradient[index.Z + inGradientOffset];

            for (int j = 0; j < info.FilterSize; j++)
            {
                for (int i = 0; i < info.FilterSize; i++)
                {
                    if (info.TryGetInputIndex(index.Z, i, j, out int inputIndex))
                    {
                        int filterIndex = info.FilterIndex(i, j, index.Y);
                        float dK = dL * input[inputIndex + inputOffset];
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
        private static void BackwardsGradientKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> filter, ArrayView<float> outGradient, ArrayView<LayerInfo> infoView)
        {
            LayerInfo info = infoView[0];
            (int outGradientOffset, int inGradientOffset) = info.GetOffset(index.X, index.Y);

            float dL = inGradient[index.Z + inGradientOffset];

            for (int j = 0; j < info.FilterSize; j++)
            {
                for (int i = 0; i < info.FilterSize; i++)
                {
                    if (info.TryGetInputIndex(index.Z, i, j, out int outGradientIndex))
                    {
                        int filterIndex = info.FilterIndex(i, j, index.Y);
                        float dP = dL * filter[filterIndex];
                        Atomic.Add(ref outGradient[outGradientIndex + outGradientOffset], dP);
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
        private static void ForwardKernel(Index3D index, ArrayView<float> input, ArrayView<float> convoluted, ArrayView<float> filter, ArrayView<LayerInfo> infoView)
        {
            LayerInfo info = infoView[0];
            (int inputOffset, int outputOffset) = info.GetOffset(index.X, index.Y);
            float sum = 0;

            for (int j = 0; j < info.FilterSize; j++)
            {
                for (int i = 0; i < info.FilterSize; i++)
                {
                    if (info.TryGetInputIndex(index.Z, i, j, out int inputIndex))
                        sum += filter[info.FilterIndex(i, j, index.Y)] * input[inputIndex + inputOffset];
                }
            }

            convoluted[index.Z + outputOffset] = sum;
        }

        /// <summary>
        /// Backpropagates through the layer without updating any of the filter weights. Called when learning rate is zero.
        /// </summary>
        private void BackwardsNoUpdate()
        {
            _buffers.OutGradient.SubView(0, _inputDimensions * _batchSize * Infos(0).InputArea).MemSetToZero();

            Index3D index = new(_batchSize, _outputDimensions, Infos(0).OutputArea);
            BackwardsOutGradientAction(index, _buffers.InGradient, _filters.WeightsGPU<float>(), _buffers.OutGradient, _deviceInfos);

            Synchronize();

            _filters.DecrementLiveWeights();
        }

        /// <summary>
        /// Perform standard backpropagation through the layer, updating it's weights. Called when learning rate is greater than 0.
        /// </summary>
        /// <param name="learningRate">The overall learning rate for the layer updates, corrected for the influence of bias in the first and second moments.</param>
        /// <param name="firstMomentDecay">The exponential decay rate for the first moment.</param>
        /// <param name="secondMomentDecay">The exponential decay rate for the second moment.</param>
        private void BackwardsUpdate(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            _buffers.OutGradient.SubView(0, _inputDimensions * _batchSize * Infos(0).InputArea).MemSetToZero();

            Index3D index = new(_batchSize, _outputDimensions, Infos(0).OutputArea);

            BackwardsOutGradientAction(index, _buffers.InGradient, _filters.WeightsGPU<float>(), _buffers.OutGradient, _deviceInfos);
            BackwardsFilterAction(index, _buffers.InGradient, _inputCopy.GetArrayView<float>(), _filters.GradientGPU<float>(), _deviceInfos);

            Index3D biasIndex = new(_outputDimensions * _outputShapes[0].Area, 1, _batchSize);
            GPUManager.AddAction(biasIndex, _bias.GradientGPU<float>(), _buffers.InGradient, _outputDimensions * _outputShapes[0].Area);

            Synchronize();

            _inputCopy.DecrementLiveCount();
            _bias.DecrementLiveGradient();
            _bias.UpdateWeights(learningRate, firstMomentDecay, secondMomentDecay);

            _filters.DecrementLiveGradient();
            _filters.DecrementLiveWeights();
            _filters.UpdateWeights(learningRate, firstMomentDecay, secondMomentDecay);
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
    }
}