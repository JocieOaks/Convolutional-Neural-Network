﻿using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.Weighted;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="AveragePool"/> class is a <see cref="Layer"/> that outputs a downscaled
    /// <see cref="FeatureMap"/> of the previous <see cref="Layer"/>'s outputs.
    /// </summary>
    [Serializable]
    public class AveragePool : Layer
    {
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo> s_backwardsAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo>(BackwardsKernel);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo> s_forwardAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo>(ForwardPoolKernel);

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
        public override void Backwards(int batchSize, bool update)
        {

            Index3D index = new(OutputShape.Area, InputShape.Dimensions, batchSize);
            s_backwardsAction(index, Buffers.InGradient, Buffers.OutGradient, Info);

            Synchronize();
        }

        /// <inheritdoc/>
        public override void Forward(int batchSize)
        {

            Index3D index = new(OutputShape.Area, InputShape.Dimensions, batchSize);
            s_forwardAction(index, Buffers.Input, Buffers.Output, Info);

            Synchronize();
        }

        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShape, PairedBuffers buffers, int maxBatchSize)
        {
            if (Ready)
                return OutputShape;
            Ready = true;

            BaseStartup(inputShape, buffers);
            
            return OutputShape;
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
        private static void BackwardsKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, LayerInfo info)
        {
            (int inputOffset, int outputOffset) = info.GetOffset(index.Y, index.Z);

            float dL = inGradient[index.X + outputOffset] * info.InverseFilterArea;

            for (int j = 0; j < info.FilterSize; j++)
            {
                for (int i = 0; i < info.FilterSize; i++)
                {
                    if (info.TryGetExpansionIndex(index.X, i, j, out int inputIndex))
                    {
                        outGradient[inputIndex + inputOffset] = dL;
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
        private static void ForwardPoolKernel(Index3D index, ArrayView<float> input, ArrayView<float> pooled, LayerInfo info)
        {
            (int inputOffset, int outputOffset) = info.GetOffset(index.Y, index.Z);

            float sum = 0;
            for (int j = 0; j < info.FilterSize; j++)
            {
                for (int i = 0; i < info.FilterSize; i++)
                {
                    if (info.TryGetExpansionIndex(index.X, i, j, out int inputIndex))
                        sum += input[inputIndex + inputOffset];
                }
            }

            pooled[index.X + outputOffset] = sum * info.InverseFilterArea;
        }

        /// <summary>
        /// Gets the <see cref="LayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="LayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="LayerInfo"/> corresponding to an input dimension.</returns>
        private LayerInfo Info => (LayerInfo)LayerInfo;
    }
}