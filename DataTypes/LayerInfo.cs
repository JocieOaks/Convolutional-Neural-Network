namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="ILayerInfo"/> interface is for structs to store a variety of data about <see cref="Layers.Layer"/>s
    /// and <see cref="FeatureMap"/>s for use by an <see cref="ILGPU"/> kernel.
    /// </summary>
    public interface ILayerInfo
    {
        /// <value>The width of the input <see cref="FeatureMap"/>.</value>
        int InputWidth { get; }

        /// <value>The length of the input <see cref="FeatureMap"/>.</value>
        int InputLength { get; }

        /// <value>The area of the input <see cref="FeatureMap"/>.</value>
        int InputArea => InputWidth * InputLength;

        /// <value>One divided by the <see cref="Layers.Layer"/>'s <see cref="FilterSize"/> squared.</value>
        float InverseKSquared { get; }

        /// <value>The length and width of the <see cref="Layers.Layer"/>'s filters.</value>
        int FilterSize { get; }

        /// <value>The width of the output <see cref="FeatureMap"/>.</value>
        int OutputWidth { get; }

        /// <value>The length of the output <see cref="FeatureMap"/>.</value>
        int OutputLength { get; }

        /// <value>The area of the output <see cref="FeatureMap"/>.</value>
        public int OutputArea => OutputWidth * OutputLength;

        /// <value>The stride of the <see cref="Layers.Layer"/>'s filter.</value>
        int Stride { get; }
    }

    /// <summary>
    /// The <see cref="LayerInfo"/> struct contains a variety of data about <see cref="Layers.Layer"/>s
    /// and <see cref="FeatureMap"/>s for use by an <see cref="ILGPU"/> kernel.
    /// </summary>
    public readonly struct LayerInfo : ILayerInfo
    {
        /// <inheritdoc/>
        public int InputWidth { get; init; }

        /// <inheritdoc/>
        public int InputLength { get; init; }

        /// <inheritdoc/>
        public int InputArea => InputWidth * InputLength;

        /// <inheritdoc/>
        public float InverseKSquared { get; init; }

        /// <inheritdoc/>
        public int FilterSize { get; init; }

        /// <inheritdoc/>
        public int OutputWidth { get; init; }

        /// <inheritdoc/>
        public int OutputLength { get; init; }

        /// <inheritdoc/>
        public int OutputArea => OutputWidth * OutputLength;

        public int InputDimensions { get; init; }

        public int OutputDimensions { get; init; }

        /// <inheritdoc/>
        public int Stride { get; init; }

        public int Padding { get; init; }

        public bool TryGetInputIndex(int outputIndex, int shiftX, int shiftY, out int index)
        {
            
            int strideY = outputIndex / OutputWidth;
            int strideX = outputIndex - (strideY * OutputWidth);

            shiftX += strideX * Stride - Padding;
            shiftY += strideY * Stride - Padding;
            index = shiftY * InputWidth + shiftX;
            return shiftX >= 0 && shiftY >= 0 && shiftX < InputWidth && shiftY < InputLength;
        }

        public (int, int) GetOffset(int batchIndex,int dimension)
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

    public readonly struct InverseLayerInfo : ILayerInfo
    {
        /// <inheritdoc/>
        public int InputWidth { get; init; }

        /// <inheritdoc/>
        public int InputLength { get; init; }

        /// <inheritdoc/>
        public int InputArea => InputWidth * InputLength;

        /// <inheritdoc/>
        public float InverseKSquared { get; init; }

        /// <inheritdoc/>
        public int FilterSize { get; init; }

        /// <inheritdoc/>
        public int OutputWidth { get; init; }

        /// <inheritdoc/>
        public int OutputLength { get; init; }

        /// <inheritdoc/>
        public int OutputArea => OutputWidth * OutputLength;

        public int InputDimensions { get; init; }

        public int OutputDimensions { get; init; }

        /// <inheritdoc/>
        public int Stride { get; init; }

        public int Padding { get; init; }

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

    /// <summary>
    /// The <see cref="StaticLayerInfo"/> struct contains a variety of data about <see cref="Layers.Layer"/>s
    /// and <see cref="FeatureMap"/>s for use by an <see cref="ILGPU"/> kernel, where the input and output <see cref="FeatureMap"/>s have
    /// the same dimensions.
    /// </summary>
    public readonly struct StaticLayerInfo : ILayerInfo
    {
        /// <value>The width of both the input and output <see cref="FeatureMap"/>s.</value>
        public int Width { get; init; }

        /// <value>The length of both the input and output <see cref="FeatureMap"/>s.</value>
        public int Length { get; init; }

        /// <value>The area of both the input and output <see cref="FeatureMap"/>s.</value>
        public int Area => Width * Length;

        /// <inheritdoc/>
        public int InputWidth => Width;

        /// <inheritdoc/>
        public int InputLength => Length;

        /// <inheritdoc/>
        public float InverseKSquared => 1;

        /// <inheritdoc/>
        public int FilterSize => 1;

        /// <inheritdoc/>
        public int OutputWidth => Width;

        /// <inheritdoc/>
        public int OutputLength => Length;

        public int OutputDimensions { get; init; }
        public int InputDimensions { get; init; }

        /// <inheritdoc/>
        public int Stride => 1;

        /// <summary>
        /// Calculates the single dimensional array index for a flattened <see cref="FeatureMap"/>.
        /// </summary>
        /// <param name="x">The x coordinate of the desired index.</param>
        /// <param name="y">The y coordinate of the desired index.</param>
        /// <returns>Returns the index corresponding to (<paramref name="x"/>, <paramref name="y"/>).</returns>
        public int Index(int x, int y)
        {
            return y * Width + x;
        }
    }
}