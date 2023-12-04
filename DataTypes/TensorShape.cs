namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="TensorShape"/> struct describes the measurements of a <see cref="Tensor"/>.
    /// </summary>
    public readonly struct TensorShape
    {
        /// <summary>
        /// Initializes a new <see cref="TensorShape"/> with the given shape.
        /// </summary>
        /// <param name="width">The width of the <see cref="TensorShape"/>.</param>
        /// <param name="length">The length of the <see cref="TensorShape"/>.</param>
        /// <param name="dimensions">The dimensions of the <see cref="TensorShape"/>.</param>
        public TensorShape(int width, int length, int dimensions)
        {
            Width = width;
            Length = length;
            Dimensions = dimensions;
        }

        /// <value>The area of each 2D element of a <see cref="Tensor"/>.</value>
        public int Area => Width * Length;

        /// <value>The number of dimensions.</value>
        public int Dimensions { get; }

        /// <value>The length of a <see cref="Tensor"/>.</value>
        public int Length { get; }

        /// <value>The total number of elements in a <see cref="Tensor"/>.</value>
        public int Volume => Area * Dimensions;

        /// <value>The width of a <see cref="Tensor"/>.</value>
        public int Width { get; }
        /// <summary>
        /// Finds the starting index for a single 2D element of multiple <see cref="Tensor"/>s concatenated into a single 1D array.
        /// </summary>
        /// <param name="tensorIndex">The index of the <see cref="Tensor"/> of interest.</param>
        /// <param name="dimension">The dimension of interest.</param>
        /// <returns>Returns the index of the 0th element of the flattened 2D array.</returns>
        public int GetOffset(int tensorIndex, int dimension)
        {
            return (tensorIndex * Dimensions + dimension) * Area;
        }

        /// <summary>
        /// Tries to find the index of within a flattened 2D array with the given <see cref="TensorShape"/>.
        /// </summary>
        /// <param name="inputIndex">The original starting index.</param>
        /// <param name="shiftX">The shift of the x-axis to the desired element.</param>
        /// <param name="shiftY">The shift of the y-axis to the desired element.</param>
        /// <param name="outIndex">The presumed index of the desired element in the flattened array.</param>
        /// <returns>Returns false if the desired index is bounds of the 2D array and thus <param name="outIndex"/> is invalid.</returns>
        public bool TryGetIndex(int inputIndex, int shiftX, int shiftY, out int outIndex)
        {

            int y = inputIndex / Width;
            int x = inputIndex - (y * Width);

            shiftX += x;
            shiftY += y;
            outIndex = shiftY * Width + shiftX;
            return shiftX >= 0 && shiftY >= 0 && shiftX < Width && shiftY < Length;
        }
    }
}
