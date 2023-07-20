using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="AveragePool"/> class is a <see cref="Layer"/> that outputs a downscaled
    /// <see cref="FeatureMap"/> of the previous <see cref="Layer"/>'s outputs.
    /// </summary>
    [Serializable]
    public class AveragePool : Layer, IStructuralLayer
    {
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>> s_backwardsAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsKernel);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>> s_forwardAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>>(ForwardKernel);
        private ArrayView<LayerInfo> _deviceInfos;

        /// <summary>
        /// Initializes a new instance of the <see cref="AveragePool"/> class.
        /// </summary>
        /// <param name="filterSize">The width and height of a block of pixels to be averaged.</param>
        public AveragePool(int filterSize) : base(filterSize, filterSize)
        {
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor]
        private AveragePool() : base()
        {
        }

        /// <inheritdoc/>
        [JsonIgnore] public override string Name => "Average Pool Layer";
        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {

            Index3D index = new(_batchSize, _inputDimensions, Infos(0).OutputArea);
            s_backwardsAction(index, _buffers.InGradient, _buffers.OutGradient, _deviceInfos);

            Synchronize();
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            
            Index3D index = new(_batchSize, _inputDimensions, Infos(0).OutputArea);
            s_forwardAction(index, _buffers.Input, _buffers.Output, _deviceInfos);

            Synchronize();
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, int batchSize)
        {
            BaseStartup(inputShapes, buffers, batchSize);
            _deviceInfos = GPU.GPUManager.Accelerator.Allocate1D(Array.ConvertAll(_layerInfos, info => (LayerInfo)info)).View;
            return _outputShapes;
        }

        /// <summary>
        /// An ILGPU kernel to calculate the gradients for backpropagating the previous layer.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
        /// <param name="inGradient">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the incoming
        /// gradient from the following <see cref="Layer"/>.</param>
        /// <param name="outGradient">An <see cref="ArrayView1D{T, TStride}"/> of floats to sum the outgoing gradient.
        /// Because <see cref="Color"/> cannot be summed atomically, every three floats represents a single
        /// <see cref="Color"/> in the gradient.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void BackwardsKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<LayerInfo> infoView)
        {
            LayerInfo info = infoView[index.Y];
            (int inputOffset, int outputOffset) = info.GetOffset(index.X, index.Y);

            for (int j = 0; j < info.FilterSize; j++)
            {
                for (int i = 0; i < info.FilterSize; i++)
                {
                    if (info.TryGetInputIndex(index.Z, i, j, out int inputIndex))
                    {
                        outGradient[inputIndex + inputOffset] = inGradient[index.Z + outputOffset];
                    }
                }
            }
        }

        /// <summary>
        /// An ILGPU kernel to calculate the downscaled version of a <see cref="FeatureMap"/>.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="pooled">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s to set for the outgoing
        /// pooled <see cref="FeatureMap"/>.</param>
        /// <param name="filter">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing one of the
        /// <see cref="Convolution"/>'s filters.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void ForwardKernel(Index3D index, ArrayView<float> input, ArrayView<float> pooled, ArrayView<LayerInfo> infoView)
        {
            LayerInfo info = infoView[index.Y];
            (int inputOffset, int outputOffset) = info.GetOffset(index.X, index.Y);

            float sum = 0;
            for (int j = 0; j < info.FilterSize; j++)
            {
                for (int i = 0; i < info.FilterSize; i++)
                {
                    if (info.TryGetInputIndex(index.Z, i, j, out int inputIndex))
                        sum += input[inputIndex + inputOffset];
                }
            }

            pooled[index.Z + outputOffset] = sum * info.InverseKSquared;
        }

        /// <summary>
        /// Gets the <see cref="LayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="LayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="LayerInfo"/> corresponding to an input dimension.</returns>
        private LayerInfo Infos(int index)
        {
            return (LayerInfo)_layerInfos[index];
        }
    }
}