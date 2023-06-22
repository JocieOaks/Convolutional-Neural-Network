using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="ColorVector"/> class stores an array of <see cref="Color"/>s for performing vector mathematics.
    /// Technically because <see cref="Color"/> is itself a vector, <see cref="ColorVector"/> is more accurately a rank two tensor. However,
    /// for simplification <see cref="Color"/> is generally treated as a scaler value.
    /// </summary>
    [Serializable]
    public class ColorVector
    {
        [JsonProperty] private readonly Color[] _values;

        /// <summary>
        /// Initializes a new <see cref="ColorVector"/> of the given length.
        /// </summary>
        /// <param name="length">The number of dimensions of the <see cref="ColorVector"/>.</param>
        public ColorVector(int length)
        {
            _values = new Color[length];
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor]
        private ColorVector()
        {
        }

        /// <summary>
        /// The number of dimensions of the <see cref="ColorVector"/>.
        /// </summary>
        [JsonIgnore] public int Length => _values.Length;

        /// <summary>
        /// Indexes the <see cref="ColorVector"/> returning the <see cref="Color"/> at a given index.
        /// </summary>
        /// <param name="index">The index of the desired <see cref="Color"/>.</param>
        /// <returns>Returns the <see cref="Color"/> at <paramref name="index"/> dimension of the <see cref="ColorVector"/>.</returns>
        public Color this[int index]
        {
            get => _values[index];
            set => _values[index] = value;
        }

        /// <summary>
        /// Multiplies a <see cref="FeatureMap"/> tensor of dimensions n x m x 3, by a <see cref="ColorVector"/> of dimensions m x 3,
        /// performing double tensor contraction to get a vector of length n.
        /// </summary>
        /// <param name="matrix">The first tensor of dimensions n x m x 3.</param>
        /// <param name="vector">The <see cref="ColorVector"/>, a  tensor of dimensions m x 3.</param>
        /// <returns>Returns a new <see cref="Vector"/> of length equal to <paramref name="matrix"/> width.</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="matrix"/> <see cref="FeatureMap.Length"/> is not equal to
        /// <paramref name="vector"/>'s length.</exception>
        public static Vector operator *(FeatureMap matrix, ColorVector vector)
        {
            if (matrix.Length != vector.Length)
                throw new ArgumentException("Matrix and vector are not compatible.");

            Vector output = new(matrix.Width);
            for (int i = 0; i < matrix.Width; i++)
            {
                for (int j = 0; j < matrix.Length; j++)
                {
                    output[i] += Color.Dot(matrix[i, j], vector[j]);
                }
            }

            return output;
        }
    }
}