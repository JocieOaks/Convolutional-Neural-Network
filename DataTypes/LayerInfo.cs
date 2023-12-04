using ConvolutionalNeuralNetwork.Layers;
using ILGPU;
using ILGPU.Algorithms;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="LayerInfo"/> struct contains a variety of data about a <see cref="Layer"/>
    /// and its input and output <see cref="Tensor"/>s for use by an <see cref="ILGPU"/> kernel.
    /// </summary>
    public readonly struct LayerInfo
    {

        /// <summary>
        /// Initializes a new instance of <see cref="Layer"/>.
        /// </summary>
        /// <param name="expansionShape">The <see cref="TensorShape"/> corresponding to the larger of the two <see cref="Tensor"/>s.
        /// May be input or output.</param>
        /// <param name="contractionShape">The <see cref="TensorShape"/> corresponding to the smaller of the two <see cref="Tensor"/>s.
        /// May be input or output.</param>
        /// <param name="filterSize">The width and length of the <see cref="Layer"/>'s filter.</param>
        /// <param name="stride">The stride of the <see cref="Layer"/>.</param>
        public LayerInfo(TensorShape expansionShape,  TensorShape contractionShape, int filterSize, int stride)
        {
            ContractionShape = contractionShape;
            ExpansionShape = expansionShape;
            FilterSize = filterSize;
            FilterArea = filterSize * filterSize;
            InverseFilterArea = 1f / FilterArea;
            Padding = (filterSize - 1) / 2;
            Stride = stride;
        }

        /// <value>The area of the <see cref="ContractionShape"/> <see cref="Tensor"/>.</value>
        public int ContractionArea => ContractionShape.Area;

        /// <value>The dimensions of the <see cref="ContractionShape"/> <see cref="Tensor"/>.</value>
        public int ContractionDimensions => ContractionShape.Dimensions;

        /// <value>The length of the <see cref="ContractionShape"/> <see cref="Tensor"/>.</value>
        public int ContractionLength => ContractionShape.Length;

        /// <value>The <see cref="TensorShape"/> referring to the smaller <see cref="Tensor"/> from the <see cref="Layer"/>s input and output.</value>
        public TensorShape ContractionShape { get; }

        /// <value>The width of the <see cref="ContractionShape"/> <see cref="Tensor"/>.</value>
        public int ContractionWidth => ContractionShape.Width;

        /// <value>The area of the <see cref="ExpansionShape"/> <see cref="Tensor"/>.</value>
        public int ExpansionArea => ExpansionShape.Area;

        /// <value>The dimensions of the <see cref="ExpansionShape"/> <see cref="Tensor"/>.</value>
        public int ExpansionDimensions => ExpansionShape.Dimensions;

        /// <value>The length of the <see cref="ExpansionShape"/> <see cref="Tensor"/>.</value>
        public int ExpansionLength => ExpansionShape.Length;

        /// <value>The <see cref="TensorShape"/> referring to the larger <see cref="Tensor"/> from the <see cref="Layer"/>s input and output.</value>
        public TensorShape ExpansionShape { get; }
        /// <value>The width of the <see cref="ExpansionShape"/> <see cref="Tensor"/>.</value>
        public int ExpansionWidth => ExpansionShape.Width;

        /// <value>The area of the <see cref="Layer"/>'s filter.</value>
        public int FilterArea { get; }

        /// <value>The length and width of the <see cref="Layer"/>'s filter.</value>
        public int FilterSize { get; }

        /// <value>One over the area of the <see cref="Layer"/>'s filter.</value>
        public float InverseFilterArea { get; }

        /// <value>The amount of padding to offset for the size of the input <see cref="Tensor"/>.</value>
        public int Padding { get; }

        /// <value>The stride of the <see cref="Layer"/>.</value>
        public int Stride { get; }

        /// <summary>
        /// Deconstructs an <see cref="ILGPU"/> kernel <see cref="Index3D"/> to get useful values.
        /// </summary>
        /// <param name="index">The <see cref="Index3D"/> being deconstructed.
        /// X: Contraction Volume
        /// Y: Expansion Dimensions
        /// Z: Batch size</param>
        /// <param name="mapIndex">The index within a flattened 2D array of the Expansion <see cref="Tensor"/>.</param>
        /// <param name="expansionOffset">The index of the zeroth element of a flattened 2D array corresponding to the specified dimension and tensor.</param>
        /// <param name="contractionIndex">The index within a flattened 2D array of the Contraction <see cref="Tensor"/>.</param>
        /// <param name="filterDimension">The corresponding filter for the specific input and output dimensions.</param>
        public void DeconstructContraction(Index3D index, out int mapIndex, out int expansionOffset,
            out int contractionIndex, out int filterDimension)
        {
            contractionIndex = index.Z * ContractionArea * ContractionDimensions + index.X;
            int contractionDimension = index.X / ContractionArea;
            filterDimension = index.Y * ContractionDimensions + contractionDimension;
            mapIndex = index.X % ContractionArea;
            expansionOffset = (index.Y + index.Z * ExpansionDimensions) * ExpansionArea;
        }

        /// <summary>
        /// Deconstructs an <see cref="ILGPU"/> kernel <see cref="Index3D"/> to get useful values.
        /// </summary>
        /// <param name="index">The <see cref="Index3D"/> being deconstructed.
        ///     X: Contraction Volume
        ///     Y: Expansion Dimensions
        ///     Z: Batch size</param>
        /// <param name="mapIndex">The index within a flattened 2D array of the Expansion <see cref="Tensor"/>.</param>
        /// <param name="contractionOffset">The index of the zeroth element of a flattened 2D array corresponding to the specified dimension and tensor</param>
        /// <param name="expansionIndex">The index within a flattened 2D array of the Expansion <see cref="Tensor"/>.</param>
        /// <param name="filterDimension">The corresponding filter for the specific input and output dimensions.</param>
        public void DeconstructExpansion(Index3D index, out int mapIndex, out int contractionOffset,
            out int expansionIndex, out int filterDimension)
        {
            expansionIndex = index.Z * ExpansionArea * ExpansionDimensions + index.X;
            int expansionDimension = index.X / ExpansionArea;
            filterDimension = expansionDimension * ContractionDimensions + index.Y;
            mapIndex = index.X % ExpansionArea;
            contractionOffset = (index.Y + index.Z * ContractionDimensions) * ContractionArea;
        }

        /// <summary>
        /// Calculates the single dimensional array index for a flattened filter.
        /// </summary>
        /// <param name="x">The x coordinate of the desired index.</param>
        /// <param name="y">The y coordinate of the desired index.</param>
        /// <param name="dimension">The dimension of the filter.</param>
        /// <returns>Returns the index corresponding to (<paramref name="x"/>, <paramref name="y"/>, <param name="dimension"/>).</returns>
        public int FilterIndex(int x, int y, int dimension)
        {
            return dimension * FilterArea + y * FilterSize + x;
        }

        /// <summary>
        /// Find the 2 dimensional coordinates corresponding to the index of the Contraction <see cref="Tensor"/>.
        /// </summary>
        /// <param name="contractionIndex">The index of the Contraction <see cref="Tensor"/>.</param>
        /// <returns>Returns the desired coordinates as a tuple.</returns>
        public (int, int) GetContractionCoordinates(int contractionIndex)
        {
            int x = contractionIndex % ContractionWidth;
            int y = contractionIndex / ContractionWidth;
            return (x, y);
        }

        /// <summary>
        /// Gets the index in the Contraction <see cref="Tensor"/> corresponding to the given index in the Expansion <see cref="Tensor"/>
        /// with some shift.
        /// </summary>
        /// <param name="expansionIndex">The original index of the Expansion <see cref="Tensor"/>.</param>
        /// <param name="shiftX">The shift in the x-axis.</param>
        /// <param name="shiftY">The shift in the y-axis.</param>
        /// <returns>Returns the desired index.</returns>
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

        /// <summary>
        /// Find the 2 dimensional coordinates corresponding to the index of the Expansion <see cref="Tensor"/>.
        /// </summary>
        /// <param name="expansionIndex">The index of the Expansion <see cref="Tensor"/>.</param>
        /// <returns>Returns the desired coordinates as a tuple.</returns>
        public (int, int) GetExpansionCoordinates(int expansionIndex)
        {
            int x = expansionIndex % ExpansionWidth;
            int y = expansionIndex / ExpansionWidth;
            return (x, y);
        }

        /// <summary>
        /// Gets the index in the Expansion <see cref="Tensor"/> corresponding to the given index in the Contraction <see cref="Tensor"/>
        /// with some shift.
        /// </summary>
        /// <param name="contractionIndex">The original index of the Contraction <see cref="Tensor"/>.</param>
        /// <param name="shiftX">The shift in the x-axis.</param>
        /// <param name="shiftY">The shift in the y-axis.</param>
        /// <returns>Returns the desired index.</returns>
        public int GetExpansionIndex(int contractionIndex, int shiftX, int shiftY)
        {
            int strideY = contractionIndex / ContractionWidth;
            int strideX = contractionIndex - (strideY * ContractionWidth);

            shiftX += strideX * Stride - Padding;
            shiftY += strideY * Stride - Padding;
            return shiftY * ExpansionWidth + shiftX;
        }

        /// <summary>
        /// Gets the index of the first element of a particular <see cref="Tensor"/> and dimension.
        /// </summary>
        /// <returns>Returns the zeroth element index.</returns>
        public (int, int) GetOffset(int dimension, int tensorIndex)
        {
            int offsetCount = tensorIndex * ExpansionDimensions + dimension;
            return (offsetCount * ExpansionArea, offsetCount * ContractionArea);
        }

        /// <summary>
        /// Tries to get the index in the Contraction <see cref="Tensor"/> corresponding to the given index in the Expansion <see cref="Tensor"/>
        /// with some shift.
        /// </summary>
        /// <param name="expansionIndex">The original index of the Expansion <see cref="Tensor"/>.</param>
        /// <param name="shiftX">The shift in the x-axis.</param>
        /// <param name="shiftY">The shift in the y-axis.</param>
        /// <param name="index">The index in the Contraction <see cref="Tensor"/>.</param>
        /// <returns>Returns false if the desired index is bounds of the 2D array and thus <param name="index"/> is invalid.</returns>
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

        /// <summary>
        /// Gets the index in the Expansion <see cref="Tensor"/> corresponding to the given index in the Contraction <see cref="Tensor"/>
        /// with some shift.
        /// </summary>
        /// <param name="contractionIndex">The original index of the Contraction <see cref="Tensor"/>.</param>
        /// <param name="shiftX">The shift in the x-axis.</param>
        /// <param name="shiftY">The shift in the y-axis.</param>
        /// <param name="index">The index in the Expansion <see cref="Tensor"/>.</param>
        /// <returns>Returns false if the desired index is bounds of the 2D array and thus <param name="index"/> is invalid.</returns>
        public bool TryGetExpansionIndex(int contractionIndex, int shiftX, int shiftY, out int index)
        {
            int strideY = contractionIndex / ContractionWidth;
            int strideX = contractionIndex - (strideY * ContractionWidth);

            shiftX += strideX * Stride - Padding;
            shiftY += strideY * Stride - Padding;
            index = shiftY * ExpansionWidth + shiftX;
            return shiftX >= 0 && shiftY >= 0 && shiftX < ExpansionWidth && shiftY < ExpansionLength;
        }
    }
}