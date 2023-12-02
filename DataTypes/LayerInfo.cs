using ILGPU.Algorithms;

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
        public LayerInfo(Shape expansionShape,  Shape contractionShape, int filterSize, int stride)
        {
            ExpansionWidth = expansionShape.Width;
            ExpansionLength = expansionShape.Length;
            ExpansionArea = expansionShape.Area;
            ExpansionDimensions = expansionShape.Dimensions;
            ContractionWidth = contractionShape.Width;
            ContractionLength = contractionShape.Length;
            ContractionArea = contractionShape.Area;
            ContractionDimensions = contractionShape.Dimensions;
            FilterSize = filterSize;
            FilterArea = filterSize * filterSize;
            InverseFilterArea = 1f / FilterArea;
            Padding = (filterSize - 1) / 2;
            Stride = stride;
        }

        /// <inheritdoc/>
        public int ExpansionWidth { get; }

        /// <inheritdoc/>
        public int ExpansionLength { get; }

        /// <inheritdoc/>
        public int ExpansionArea { get; }

        /// <inheritdoc/>
        public float InverseFilterArea { get; }

        /// <inheritdoc/>
        public int FilterSize { get; }

        public int FilterArea { get; }

        /// <inheritdoc/>
        public int ContractionWidth { get; }

        /// <inheritdoc/>
        public int ContractionLength { get; }

        /// <inheritdoc/>
        public int ContractionArea { get; }

        public int ExpansionDimensions { get; }

        public int ContractionDimensions { get; }

        /// <inheritdoc/>
        public int Stride { get; }

        public int Padding { get; }

        public bool TryGetExpansionIndex(int contractionIndex, int shiftX, int shiftY, out int index)
        {
            int strideY = contractionIndex / ContractionWidth;
            int strideX = contractionIndex - (strideY * ContractionWidth);

            shiftX += strideX * Stride - Padding;
            shiftY += strideY * Stride - Padding;
            index = shiftY * ExpansionWidth + shiftX;
            return shiftX >= 0 && shiftY >= 0 && shiftX < ExpansionWidth && shiftY < ExpansionLength;
        }

        public bool TryGetContractionIndex(int expansionIndex, int shiftX, int shiftY, out int index)
        {
            int x = expansionIndex % ExpansionWidth;
            int y = expansionIndex / ExpansionWidth;

            x += Padding - shiftX;
            y += Padding - shiftY;

            x /= Stride;
            y /= Stride;



            x = XMath.Clamp(x, 0, ContractionWidth);
            y = XMath.Clamp(y, 0, ContractionLength);

            index = y * ContractionWidth + x;
            return shiftX >= 0 && shiftY >= 0 && shiftX < ContractionWidth && shiftY < ContractionLength;
        }

        public int GetExpansionIndex(int contractionIndex, int shiftX, int shiftY)
        {
            int strideY = contractionIndex / ContractionWidth;
            int strideX = contractionIndex - (strideY * ContractionWidth);

            shiftX += strideX * Stride - Padding;
            shiftY += strideY * Stride - Padding;
            return shiftY * ExpansionWidth + shiftX;
        }

        public int GetContractionIndex(int expansionIndex, int shiftX, int shiftY)
        {
            int x = expansionIndex % ExpansionWidth;
            int y = expansionIndex / ExpansionWidth;

            x += Padding - shiftX;
            y += Padding - shiftY;

            x /= Stride;
            y /= Stride;



            x = XMath.Clamp(x, 0, ContractionWidth);
            y = XMath.Clamp(y, 0, ContractionLength);

            return y * ContractionWidth + x;
        }

        public (int, int) GetExpansionCoordinates(int expansionIndex)
        {
            int x = expansionIndex % ExpansionWidth;
            int y = expansionIndex / ExpansionWidth;
            return (x, y);
        }

        public (int, int) GetContractionCoordinates(int contractionIndex)
        {
            int x = contractionIndex % ContractionWidth;
            int y = contractionIndex / ContractionWidth;
            return (x, y);
        }

        public (int,int) GetOffset(int dimension, int batchIndex)
        {
            int offsetCount = batchIndex * ExpansionDimensions + dimension;
            return (offsetCount * ExpansionArea, offsetCount * ContractionArea);
        }

        public void DeconstructContraction(int x, int y, int z, out int mapIndex, out int expansionOffset, out int contractionIndex, out int dimension)
        {
            contractionIndex = z * ContractionArea * ContractionDimensions + x;
            int contractionDimension = x / ContractionArea;
            dimension = y * ContractionDimensions + contractionDimension;
            mapIndex = x % ContractionArea;
            expansionOffset = (y + z * ExpansionDimensions) * ExpansionArea;
        }

        public void DeconstructExpansion(int x, int y, int z, out int mapIndex, out int contractionOffset, out int expansionIndex, out int dimension)
        {
            expansionIndex = z * ExpansionArea * ExpansionDimensions + x;
            int expansionDimension = x / ExpansionArea;
            dimension = expansionDimension * ContractionDimensions + y;
            mapIndex = x % ExpansionArea;
            contractionOffset = (y + z * ContractionDimensions) * ContractionArea;
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

        public (int, int) GetInputCoordinates(int outputIndex, out float xFloat, out float yFloat)
        {
            int x = outputIndex % OutputWidth;
            int y = outputIndex / OutputWidth;

            x += Padding;
            y += Padding;

            xFloat = (float)x / Stride;
            yFloat = (float)y / Stride;

            x /= Stride;
            y /= Stride;

            return (x, y);
        }

        public bool TryGetInputIndex(int x, int y, int shiftX, int shiftY, out int inputIndex)
        {
            x += shiftX;
            y += shiftY;

            inputIndex = y * InputWidth + x;

            return x >= 0 && y >= 0 && x < InputWidth && y < InputLength;
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