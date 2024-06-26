﻿using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU.Runtime;
using ILGPU;
using ILGPU.Algorithms;

namespace ConvolutionalNeuralNetwork.Layers.Weighted
{
    /// <summary>
    /// The <see cref="TransposeConvolution"/> class is a <see cref="Layer"/> that performs a 2D Convolution,
    /// by passing a 2D filter over a <see cref="Tensor"/>, performing the inverse action of a <see cref="Convolution"/> layer.
    /// </summary>
    public class TransposeConvolution : WeightedLayer
    {
        private static readonly Action<KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> s_backwardsFilterAction = GPUManager.Accelerator.LoadStreamKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(TransConvFilterKernel);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> s_backwardsOutGradientAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(TransConvGradientKernel);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo> s_forwardAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, LayerInfo>(TransConvKernel);
        private readonly int _outputDimensions;

        /// <summary>
        /// Initializes a new instance of the <see cref="TransposeConvolution"/> layer.
        /// </summary>
        /// <param name="filterSize">The width and height of a filter.</param>
        /// <param name="stride">The amount of movement over the image for each filter pass.</param>
        /// <param name="outputDimensions">A factor relating the number of input layers to the number of output layers.
        /// Must be positive. To reduce the number of output dimensions, use a <see cref="Summation"/> layer afterwards.</param>
        /// <param name="weights">The initial weights for the <see cref="Layer"/>'s filters.</param>
        /// <param name="bias">The initial weights for the <see cref="Layer"/>'s bias.</param>
        public TransposeConvolution(int filterSize, int stride, int outputDimensions, Weights weights, Weights bias) : base(filterSize, stride, weights, bias)
        {
            if (outputDimensions < 1)
            {
                throw new ArgumentException("Dimension multiplier must be greater than or equal to 1.");
            }
            _outputDimensions = outputDimensions;
        }

        /// <inheritdoc/>
        public override string Name => "Transpose Convolutional Layer";

        private LayerInfo Info => LayerInfo;
        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int maxBatchSize)
        {
            if (Initialized)
                return OutputShape;
            Initialized = true;

            if (Weights == null)
            {
                BaseStartup(inputShape, views, _outputDimensions);
            }
            else
            {
                BaseStartup(inputShape, views, Weights.Length / FilterSize / FilterSize / inputShape.Dimensions);
            }

            InputCopy = new Vector(InputShape.Dimensions * maxBatchSize * InputShape.Area);

            return OutputShape;
        }

        /// <inheritdoc />
        protected override void BackwardsNoUpdate(int batchSize)
        {
            Views.OutGradient.SubView(0, batchSize * InputShape.Volume).MemSetToZero();

            Index3D index = new(InputShape.Volume, OutputShape.Dimensions, batchSize);
            s_backwardsOutGradientAction(index, Views.InGradient, Views.OutGradient, Weights.WeightsView(), Info);
        }

        /// <inheritdoc />
        protected override void BackwardsUpdate(int batchSize)
        {

            Views.OutGradient.SubView(0, batchSize * InputShape.Volume).MemSetToZero();

            Index3D index = new(InputShape.Volume, OutputShape.Dimensions, batchSize);
            s_backwardsOutGradientAction(index, Views.InGradient, Views.OutGradient, Weights.WeightsView(), Info);
            KernelConfig config = new(new Index3D(Info.FilterArea, OutputShape.Dimensions, batchSize), new Index3D(InputShape.Dimensions, 1, 1));
            s_backwardsFilterAction(config, Views.InGradient, InputCopy.GetArrayView(), Weights.GradientView(), Info);
        }

        /// <summary>
        /// Initializes the <see cref="Layer"/> and many of its fields.
        /// </summary>
        /// <param name="inputShape">The <see cref="TensorShape"/> of the previous <see cref="Layer"/>'s output.</param>
        /// <param name="views">The <see cref="PairedGPUViews"/> containing the input and output views.</param>
        /// <param name="outputDimensions">A factor relating the number of input layers to the number of output layers.
        /// A positive number multiplies the number of input dimensions. A negative number divides the number of dimensions.</param>
        /// <exception cref="ArgumentException">Thrown if the ratio of input layers and output layers is not an integer.</exception>
        protected new void BaseStartup(TensorShape inputShape, PairedGPUViews views, int outputDimensions = 1)
        {
            InputShape = inputShape;
            OutputShape = new TensorShape(inputShape.Width * Stride, inputShape.Length * Stride, outputDimensions);

            LayerInfo = new LayerInfo(OutputShape, inputShape, FilterSize, Stride);

            Views = views;
            views.OutputDimensionArea(OutputShape.Volume);
        }

        /// <inheritdoc/>
        protected override void ForwardChild(int batchSize)
        {

            Index1D copyIndex = new(batchSize * InputShape.Volume);
            GPUManager.CopyAction(copyIndex, Views.Input, InputCopy.GetArrayViewEmpty());

            Views.Output.SubView(0, batchSize * OutputShape.Volume).MemSetToZero();

            Index3D index = new(OutputShape.Volume, InputShape.Dimensions, batchSize);
            s_forwardAction(index, Views.Input, Views.Output, Weights.WeightsView(), Info);
        }
        private static void TransConvFilterKernel(ArrayView<float> inGradient, ArrayView<float> input, ArrayView<float> filterGradient, LayerInfo info)
        {

            int inputOffset = (Grid.IdxZ * info.ContractionDimensions + Group.IdxX) * info.ContractionArea;
            int outputOffset = (Grid.IdxZ * info.ExpansionDimensions + Grid.IdxY) * info.ExpansionArea;
            int dimension = Grid.IdxY * info.ContractionDimensions + Group.IdxX;
            float dK = 0;

            int x = Grid.IdxX % info.FilterSize;
            int y = Grid.IdxX / info.FilterSize;

            for (int i = 0; i < info.ContractionArea; i++)
            {
                if (info.TryGetContractionIndex(i, x, y, out int inputIndex))
                {
                    float dL = inGradient[outputOffset + i];
                    dK += dL * input[inputIndex + inputOffset];
                }
            }

            int filterIndex = info.FilterIndex(x, y, dimension);
            Atomic.Add(ref filterGradient[filterIndex], dK);
        }

        private static void TransConvGradientKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<float> filter, LayerInfo info)
        {
            info.DeconstructContraction(index, out int mapIndex, out int inGradientOffset, out int outGradientIndex, out int dimension);

            float sum = 0;

            (int x, int y) = info.GetContractionCoordinates(mapIndex);

            int minX = x * info.Stride - info.Padding;
            int minY = y * info.Stride - info.Padding;

            int x0 = XMath.Max(0, -minX);
            int x1 = XMath.Min(info.FilterSize, info.ExpansionWidth - minX);
            int y0 = XMath.Max(0, -minY);
            int y1 = XMath.Min(info.FilterSize, info.ExpansionLength - minY);

            for (int j = y0; j < y1; j++)
            {
                for (int i = x0; i < x1; i++)
                {
                    int inGradientIndex = info.GetExpansionIndex(mapIndex, i, j);
                    sum += filter[info.FilterIndex(i, j, dimension)] * inGradient[inGradientIndex + inGradientOffset];
                }
            }

            Atomic.Add(ref outGradient[outGradientIndex], sum);
        }

        private static void TransConvKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, ArrayView<float> filter, LayerInfo info)
        {
            info.DeconstructExpansion(index, out int mapIndex, out int inputOffset, out int outputIndex, out int dimension);

            (int x, int y) = info.GetExpansionCoordinates(mapIndex);

            int minX = x - info.FilterSize + info.Padding + 1;
            int minY = y - info.FilterSize + info.Padding + 1;
            int maxX = x + info.Padding + 1;
            int maxY = y + info.Padding + 1;

            int shiftX = minX % info.Stride;
            int shiftY = minY % info.Stride;

            shiftX -= XMath.Clamp(shiftX, 0, 1) * info.Stride;
            shiftY -= XMath.Clamp(shiftY, 0, 1) * info.Stride;

            int x0 = XMath.Max(0, maxX - info.ExpansionWidth);
            int x1 = XMath.Min(info.FilterSize + shiftX, info.FilterSize + minX);
            int y0 = XMath.Max(0, maxY - info.ExpansionLength);
            int y1 = XMath.Min(info.FilterSize + shiftY, info.FilterSize + minY);

            float sum = 0;

            for (int j = y1 - 1; j >= y0; j -= info.Stride)
            {
                for (int i = x1 - 1; i >= x0; i -= info.Stride)
                {
                    int inputIndex = info.GetContractionIndex(mapIndex, i, j);
                    int filterIndex = info.FilterIndex(i, j, dimension);
                    sum += input[inputIndex + inputOffset] * filter[filterIndex];
                }
            }

            Atomic.Add(ref output[outputIndex], sum);
        }
    }
}