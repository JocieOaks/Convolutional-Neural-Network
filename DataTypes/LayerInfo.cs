using ILGPU;
using ILGPU.IR.Values;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="ILayerInfo"/> interface is for structs to store a variety of data about <see cref="Layers.Layer"/>s
    /// and <see cref="FeatureMap"/>s for use by an <see cref="ILGPU"/> kernel.
    /// </summary>
    public interface ILayerInfo
    {
    }

    /// <summary>
    /// The <see cref="LayerInfo"/> struct contains a variety of data about <see cref="Layers.Layer"/>s
    /// and <see cref="FeatureMap"/>s for use by an <see cref="ILGPU"/> kernel.
    /// </summary>
    public readonly struct LayerInfo : ILayerInfo
    {
        public LayerInfo(Shape inputShape,  Shape outputShape, int filterSize, int stride)
        {
            InputWidth = inputShape.Width;
            InputLength = inputShape.Length;
            InputArea = inputShape.Area;
            InputDimensions = inputShape.Dimensions;
            OutputWidth = outputShape.Width;
            OutputLength = outputShape.Length;
            OutputArea = outputShape.Area;
            OutputDimensions = outputShape.Dimensions;
            FilterSize = filterSize;
            FilterArea = filterSize * filterSize;
            InverseFilterArea = 1f / FilterArea;
            Padding = (filterSize - 1) / 2;
            Stride = stride;
        }

        /// <inheritdoc/>
        public int InputWidth { get; }

        /// <inheritdoc/>
        public int InputLength { get; }

        /// <inheritdoc/>
        public int InputArea { get; }

        /// <inheritdoc/>
        public float InverseFilterArea { get; }

        /// <inheritdoc/>
        public int FilterSize { get; }

        public int FilterArea { get; }

        /// <inheritdoc/>
        public int OutputWidth { get; }

        /// <inheritdoc/>
        public int OutputLength { get; }

        /// <inheritdoc/>
        public int OutputArea { get; }

        public int InputDimensions { get; }

        public int OutputDimensions { get; }

        /// <inheritdoc/>
        public int Stride { get; }

        public int Padding { get; }

        public bool TryGetInputIndex(int outputIndex, int shiftX, int shiftY, out int index)
        {
            int strideY = outputIndex / OutputWidth;
            int strideX = outputIndex - (strideY * OutputWidth);

            shiftX += strideX * Stride - Padding;
            shiftY += strideY * Stride - Padding;
            index = shiftY * InputWidth + shiftX;
            return shiftX >= 0 && shiftY >= 0 && shiftX < InputWidth && shiftY < InputLength;
        }

        public bool TryGetInputIndex(int strideX, int strideY, int shiftX, int shiftY, out int index)
        {
            shiftX += strideX * Stride - Padding;
            shiftY += strideY * Stride - Padding;
            index = shiftY * InputWidth + shiftX;
            return shiftX >= 0 && shiftY >= 0 && shiftX < InputWidth && shiftY < InputLength;
        }

        public int OutputIndex(int x, int y)
        {
            return y * OutputWidth + x;
        }

        public (int,int) GetOffset(int dimension, int batchIndex)
        {
            int offsetCount = batchIndex * InputDimensions + dimension;
            return (offsetCount * InputArea, offsetCount * OutputArea);
        }

        public (int, int) GetOffset(int inputDimension, int outputDimension, int batchIndex)
        {
            return ((inputDimension + batchIndex * InputDimensions) * InputArea, (outputDimension + batchIndex * OutputDimensions) * OutputArea);
        }

        public void Deconstruct(int x, int y, int z, out int mapIndex, out int inputOffset, out int outputIndex, out int dimension)
        {
            outputIndex = z * OutputArea * OutputDimensions + x;
            int outputDimension = x / OutputArea;
            dimension = y * OutputDimensions + outputDimension;
            mapIndex = x % OutputArea;
            inputOffset = (y + z * InputDimensions) * InputArea;
        }

        public void Deconstruct(Index3D grid, Index3D group, out int inputOffset, out int dimension)
        {
            dimension = grid.Y * InputDimensions + group.X;
            inputOffset = (group.X + grid.Z * InputDimensions) * InputArea;
        }

        /// <summary>
        /// Calculates the single dimensional array index for a flattened filter.
        /// </summary>
        /// <param name="x">The x coordinate of the desired index.</param>
        /// <param name="y">The y coordinate of the desired index.</param>
        /// <returns>Returns the index corresponding to (<paramref name="x"/>, <paramref name="y"/>).</returns>
        public int FilterIndex(int x, int y, int dimension)
        {
            return dimension * FilterArea + y * FilterSize + x;
        }
    }

    public readonly struct InverseLayerInfo : ILayerInfo
    {
        public InverseLayerInfo(Shape inputShape, Shape outputShape, int filterSize, int stride)
        {
            InputWidth = inputShape.Width;
            InputLength = inputShape.Length;
            InputArea = inputShape.Area;
            InputDimensions = inputShape.Dimensions;
            OutputWidth = outputShape.Width;
            OutputLength = outputShape.Length;
            OutputArea = outputShape.Area;
            OutputDimensions = outputShape.Dimensions;
            FilterSize = filterSize;
            FilterArea = filterSize * filterSize;
            InverseFilterArea = 1f / FilterArea;
            Padding = (filterSize - 1) / 2;
            Stride = stride;
        }

        /// <inheritdoc/>
        public int InputWidth { get; }

        /// <inheritdoc/>
        public int InputLength { get; }

        /// <inheritdoc/>
        public int InputArea { get; }

        /// <inheritdoc/>
        public float InverseFilterArea { get; }

        /// <inheritdoc/>
        public int FilterSize { get; }

        public int FilterArea { get; }

        /// <inheritdoc/>
        public int OutputWidth { get; }

        /// <inheritdoc/>
        public int OutputLength { get; }

        /// <inheritdoc/>
        public int OutputArea { get; }

        public int InputDimensions { get; }

        public int OutputDimensions { get; }

        /// <inheritdoc/>
        public int Stride { get; }

        public int Padding { get; }

        public bool TryGetOutputIndex(int inputIndex, int shiftX, int shiftY, out int index)
        {

            int strideY = inputIndex / InputWidth;
            int strideX = inputIndex - (strideY * InputWidth);

            shiftX += strideX * Stride - Padding;
            shiftY += strideY * Stride - Padding;
            index = shiftY * OutputWidth + shiftX;
            return shiftX >= 0 && shiftY >= 0 && shiftX < OutputWidth && shiftY < OutputLength;
        }

        public (int, int) GetOffset(int batchIndex, int dimension)
        {
            return ((dimension % InputDimensions + batchIndex * InputDimensions) * InputArea, (dimension % OutputDimensions + batchIndex * OutputDimensions) * OutputArea);
        }

        /// <summary>
        /// Calculates the single dimensional array index for a flattened filter.
        /// </summary>
        /// <param name="x">The x coordinate of the desired index.</param>
        /// <param name="y">The y coordinate of the desired index.</param>
        /// <returns>Returns the index corresponding to (<paramref name="x"/>, <paramref name="y"/>).</returns>
        public int FilterIndex(int x, int y, int dimension)
        {
            return dimension * FilterSize * FilterSize + y * FilterSize + x;
        }
    }
}