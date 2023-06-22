namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="Vector"/> class stores an array of floats for performing vector mathematics.
    /// </summary>
    public class Vector
    {
        private readonly float[] _values;

        /// <summary>
        /// Initializes a new <see cref="Vector"/> using an array of floats.
        /// </summary>
        /// <param name="values">The values of the <see cref="Vector"/>.</param>
        public Vector(float[] values)
        {
            _values = values;
        }

        /// <summary>
        /// Initializes a new empty <see cref="Vector"/> of a given length.
        /// </summary>
        /// <param name="length">The number of dimensions of the <see cref="Vector"/>.</param>
        public Vector(int length)
        {
            _values = new float[length];
        }

        /// <value>The number of dimensions of the <see cref="Vector"/>.</value>
        public int Length => _values.Length;

        /// <value>The magnitude of the <see cref="Vector"/>.</value>
        public float Magnitude
        {
            get
            {
                float sum = 0;
                for (int i = 0; i < _values.Length; i++)
                {
                    sum += _values[i] * _values[i];
                }

                return MathF.Sqrt(sum);
            }
        }

        /// <summary>
        /// Indexes the <see cref="Vector"/> retrieving the value at the desired index.
        /// </summary>
        /// <param name="index">The index of the desired float.</param>
        /// <returns>Returns the float at <paramref name="index"/> dimension of the <see cref="Vector"/>.</returns>
        public float this[int index]
        {
            get { return _values[index]; }
            set { _values[index] = value; }
        }

        /// <summary>
        /// Calculates the dot product of two <see cref="Vector"/>s.
        /// </summary>
        /// <param name="v1">The first <see cref="Vector"/>.</param>
        /// <param name="v2">The second <see cref="Vector"/>.</param>
        /// <returns>Returns the dot product of <paramref name="v1"/> and <paramref name="v2"/>.</returns>
        /// <exception cref="ArgumentException"></exception>
        public static float Dot(Vector v1, Vector v2)
        {
            if (v1.Length != v2.Length)
            {
                throw new ArgumentException("Vector's not the same length.");
            }

            float dot = 0;

            for (int i = 0; i < v1.Length; i++)
            {
                dot += v1[i] * v2[i];
            }

            return dot;
        }

        /// <summary>
        /// Multiplies the vector by a scaler.
        /// </summary>
        /// <param name="vector">The <see cref="Vector"/> being multiplied.</param>
        /// <param name="scaler">The scaler factor.</param>
        /// <returns>Returns a new <see cref="Vector"/> that is parallel to <paramref name="vector"/> but
        /// whose magnitude is multiplied by <paramref name="scaler"/>.</returns>
        public static Vector operator *(Vector vector, float scaler)
        {
            float[] values = new float[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                values[i] = vector[i] * scaler;
            }
            return new Vector(values);
        }

        /// <summary>
        /// Multiplies a <see cref="Vector"/> and a matrix represented by a <see cref="FeatureMap"/>. This is functionally matrix multiplication
        /// where the <see cref="Vector"/> is a 1 x n matrix and the <see cref="FeatureMap"/> is an n x m matrix, resulting in a 1 x m vector.
        /// (However, this only holds if you treat <see cref="Color"/> as a scaler. Since the "matrix" is actually a 3 dimensional tensor,
        /// this is more accurately a tensor contraction resulting in a 1 x m x 3 tensor. However, throughout the project, <see cref="Color"/> is
        /// primarily treated as a scaler quantity for simplification)
        /// </summary>
        /// <param name="vector">The <see cref="Vector"/> of length n.</param>
        /// <param name="matrix">The <see cref="FeatureMap"/> of dimensions n x m.</param>
        /// <returns>Returns a <see cref="ColorVector"/> of length m, where m is the <see cref="FeatureMap.Length"/> of <paramref name="matrix"/>.</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="vector"/> length is not equal to <paramref name="matrix"/> width.</exception>
        public static ColorVector operator *(Vector vector, FeatureMap matrix)
        {
            if (matrix.Width != vector.Length)
                throw new ArgumentException("Matrix and vector are not compatible.");
            ColorVector output = new(matrix.Length);
            for (int i = 0; i < matrix.Width; i++)
            {
                for (int j = 0; j < matrix.Length; j++)
                {
                    output[j] += matrix[i, j] * vector[i];
                }
            }

            return output;
        }

        /// <summary>
        /// Multiplies the vector by a scaler.
        /// </summary>
        /// <param name="scaler">The scaler factor.</param>
        /// <param name="vector">The <see cref="Vector"/> being multiplied.</param>
        /// <returns>Returns a new <see cref="Vector"/> that is parallel to <paramref name="vector"/> but
        /// whose magnitude is multiplied by <paramref name="scaler"/>.</returns>
        public static Vector operator *(float scaler, Vector vector)
        {
            return vector * scaler;
        }

        /// <summary>
        /// Sums two <see cref="Vector"/>s.
        /// </summary>
        /// <param name="v1">The first <see cref="Vector"/>.</param>
        /// <param name="v2">The second <see cref="Vector"/>.</param>
        /// <returns>Returns a new <see cref="Vector"/> that is the sum of <paramref name="v1"/> and <paramref name="v2"/>.</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="v1"/> and <paramref name="v2"/> are not the same
        /// length and thus cannot be added.</exception>
        public static Vector operator +(Vector v1, Vector v2)
        {
            if (v1.Length != v2.Length)
            {
                throw new ArgumentException("Vector's not the same length.");
            }

            float[] values = new float[v1.Length];

            for (int i = 0; i < v1.Length; i++)
            {
                values[i] = v1[i] + v2[i];
            }
            return new Vector(values);
        }

        /// <summary>
        /// Normalizes the <see cref="Vector"/> returning a new unit <see cref="Vector"/> that is parralel to the original <see cref="Vector"/>.
        /// </summary>
        /// <returns>Returns a unit <see cref="Vector"/> with magnitude one that is parralel to the original <see cref="Vector"/>.</returns>
        public Vector Normalized()
        {
            float magnitude = Magnitude;

            if (magnitude == 0)
                return this;

            return this * (1 / magnitude);
        }
    }
}