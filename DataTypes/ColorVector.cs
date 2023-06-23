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
    }
}