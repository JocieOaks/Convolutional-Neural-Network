namespace ConvolutionalNeuralNetwork.DataTypes
{
    public class Vector
    {
        private readonly float[] _values;

        public float this[int index]
        {
            get { return _values[index]; }
            set { _values[index] = value; }
        }

        public Vector(float[] values)
        {
            _values = values;
        }

        public Vector(int[] values)
        {
            _values = new float[values.Length];
            for (int i = 0; i < values.Length; i++)
                _values[i] = values[i];
        }

        public Vector(int length)
        {
            _values = new float[length];
        }

        public int Length => _values.Length;

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

        public Vector Normalized()
        {
            float magnitude = Magnitude;

            if (magnitude == 0)
                return this;

            return this * (1 / magnitude);
        }

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

        public static Vector operator *(Vector vector, float mult)
        {
            float[] values = new float[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                values[i] = vector[i] * mult;
            }
            return new Vector(values);
        }

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

        public static Vector operator *(float mult, Vector vector)
        {
            float[] values = new float[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                values[i] = vector[i] * mult;
            }
            return new Vector(values);
        }
    }
}