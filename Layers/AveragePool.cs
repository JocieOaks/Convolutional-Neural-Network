﻿using ConvolutionalNeuralNetwork.DataTypes;
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
    public class AveragePool : Layer, IStructuralLayer
    {
        private MemoryBuffer1D<LayerInfo, Stride1D.Dense>[] _deviceInfos;

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
            Accelerator accelerator = Utility.Accelerator;

            var backwardsKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = accelerator.Allocate1D(new LayerInfo[] { Infos(i) });
                Index3D index = new(Infos(i).InputWidth, Infos(i).InputLength, 3);
                for (int j = 0; j < _batchSize; j++)
                {

                    backwardsKernal(index, _buffers.InGradientsFloat[i, j], _buffers.OutGradientsFloat[i, j], _deviceInfos[i].View);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            Accelerator accelerator = Utility.Accelerator;

            var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<LayerInfo>>(ForwardKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = accelerator.Allocate1D(new LayerInfo[] { Infos(i) });
                for (int j = 0; j < _batchSize; j++)
                {

                    Index2D index = new(_outputs[i, j].Width, _outputs[i, j].Length);

                    forwardKernal(index, _buffers.InputsColor[i, j], _buffers.OutputsColor[i, j], _deviceInfos[i].View);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override FeatureMap[,] Startup(FeatureMap[,] input, IOBuffers buffers)
        {
            BaseStartup(input, buffers);
            _deviceInfos = new MemoryBuffer1D<LayerInfo, Stride1D.Dense>[_inputDimensions];
            return _outputs;
        }

        /// <summary>
        /// An ILGPU kernal to calculate the gradients for backpropagating the previous layer.
        /// </summary>
        /// <param name="index">The index of the current kernal calculation to be made.</param>
        /// <param name="inGradient">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the incoming
        /// gradient from the following <see cref="Layer"/>.</param>
        /// <param name="outGradient">An <see cref="ArrayView1D{T, TStride}"/> of floats to sum the outgoing gradient.
        /// Because <see cref="Color"/> cannot be summed atomically, every three floats represents a single
        /// <see cref="Color"/> in the gradient.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void BackwardsKernal(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<LayerInfo> info)
        {
            //Unlike other Backwards Kernals, this kernal indexes by the outGradient rather than the inGradient, so the equations for index are inverted.
            int inGradientIndex = index.Y / info[0].FilterSize * info[0].OutputWidth + index.X / info[0].FilterSize;
            int outGradientIndex = index.Y * info[0].InputWidth + index.X;
            outGradient[3 * outGradientIndex + index.Z] = inGradient[3 * inGradientIndex + index.Z] * info[0].InverseKSquared;
        }

        /// <summary>
        /// An ILGPU kernal to calculate the downscaled version of a <see cref="FeatureMap"/>.
        /// </summary>
        /// <param name="index">The index of the current kernal calculation to be made.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="pooled">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s to set for the outgoing
        /// pooled <see cref="FeatureMap"/>.</param>
        /// <param name="filter">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing one of the
        /// <see cref="Convolution"/>'s filters.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> pooled, ArrayView<LayerInfo> info)
        {
            Color sum = new();
            for (int j = 0; j < info[0].FilterSize; j++)
            {
                for (int i = 0; i < info[0].FilterSize; i++)
                {
                    if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                        sum += input[inputIndex];
                }
            }
            pooled[info[0].OutputIndex(index.X, index.Y)] = sum * info[0].InverseKSquared;
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