﻿using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="Vector"/> class stores a cacheable array of floats for performing vector mathematics.
    /// </summary>
    [Serializable]
    public class Vector : Cacheable<float>, IEquatable<Vector>
    {
        [JsonProperty] private readonly float[] _values;

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

        /// <summary>
        /// Initializes a new instance of the <see cref="Vector"/> class used for deserialization.
        /// </summary>
        [JsonConstructor] Vector() { }

        /// <value>The number of dimensions of the <see cref="Vector"/>.</value>
        [JsonIgnore] public int Length => _values.Length;

        /// <value>The magnitude of the <see cref="Vector"/>.</value>
        [JsonIgnore] public float Magnitude
        {
            get
            {
                float sum = 0;
                foreach (float value in _values)
                {
                    sum += value * value;
                }

                return MathF.Sqrt(sum);
            }
        }

        /// <inheritdoc/>
        [JsonIgnore] public override long MemorySize => Length * 4;

        /// <summary>
        /// Indexes the <see cref="Vector"/> retrieving the value at the desired index.
        /// </summary>
        /// <param name="index">The index of the desired float.</param>
        /// <returns>Returns the float at <paramref name="index"/> dimension of the <see cref="Vector"/>.</returns>
        public float this[int index]
        {
            get => _values[index];
            set => _values[index] = value;
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

        ///<inheritdoc/>
        public bool Equals(Vector vector)
        {
            if (vector == null || Length != vector.Length)
                return false;

            for(int i = 0; i < Length; i++)
            {
                if (Math.Abs(_values[i] - vector[i]) > 0.0001)
                    return false;
            }
            return true;
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
        /// Calculates the distance between two vectors.
        /// </summary>
        /// <returns>Returns the distance between <param name="v1"/> and <param name="v2"/>.</returns>
        public static float Distance(Vector v1, Vector v2)
        {
            
            return MathF.Sqrt(DistanceSquared(v1, v2));
        }

        /// <summary>
        /// Calculates the squared distance between two vectors.
        /// </summary>
        /// <returns>Returns the squared distance between <param name="v1"/> and <param name="v2"/>.</returns>
        public static float DistanceSquared(Vector v1, Vector v2)
        {
            if (v1.Length != v2.Length)
                throw new ArgumentException("Vectors are not of equal length.");
            float squareDistance = 0;

            for(int i = 0; i < v1.Length; i++)
            {
                squareDistance += MathF.Pow(v1[i] - v2[i], 2);
            }

            return squareDistance;
        }


        /// <summary>
        /// Copies the <see cref="Vector"/> to an <see cref="ArrayView{T}"/>
        /// </summary>
        public void CopyToView(ArrayView<float> view)
        {
            view.SubView(0, Length).CopyFromCPU(_values);
        }

        /// <summary>
        /// Gets the <see cref="ArrayView{T}"/> for the cached <see cref="Vector"/> or allocates it if the <see cref="Vector"/> is decached.
        /// </summary>
        /// <returns>Returns an <see cref="ArrayView{T}"/>.</returns>
        public ArrayView<float> GetArrayView()
        {
            Live();
            MemoryBuffer buffer = GetBuffer();
            if (buffer == null)
            {
                (ID, buffer) = GPUManager.Allocate(this);
            }
            int bytes = Interop.SizeOf<float>();
            return new ArrayView<float>(buffer, 0, 4 * Length / bytes);
        }

        /// <summary> Gets the <see cref="ArrayView{T}"/> for the cached <see cref="Vector"/> or allocates it if the <see cref="Vector"/> is decached.
        /// A newly allocated <see cref="ArrayView{T}"/> will have random values, rather than the values of the <see cref="Vector"/>.
        /// </summary>
        /// <returns>Returns an <see cref="ArrayView{T}"/>.</returns>
        public ArrayView<float> GetArrayViewEmpty()
        {
            Live();
            MemoryBuffer buffer = GetBuffer();
            if (buffer == null)
            {
                (ID, buffer) = GPUManager.AllocateEmpty<float>(this, Length);
            }
            int bytes = Interop.SizeOf<float>();
            return new ArrayView<float>(buffer, 0, 4 * Length / bytes);
        }

        /// <summary>
        /// Gets the <see cref="ArrayView{T}"/> for the cached <see cref="Vector"/> or allocates it if the <see cref="Vector"/> is decached
        /// and sets every value to zero.
        /// </summary>
        /// <returns>Returns an <see cref="ArrayView{T}"/>.</returns>
        public ArrayView<float> GetArrayViewZeroed()
        {
            ArrayView<float> arrayView = GetArrayView();
            arrayView.MemSetToZero();
            return arrayView;
        }

        /// <inheritdoc />
        public override float[] GetValues()
        {
            return _values;
        }

        /// <summary>
        /// Normalizes the <see cref="Vector"/> returning a new unit <see cref="Vector"/> that is parallel to the original <see cref="Vector"/>.
        /// </summary>
        /// <returns>Returns a unit <see cref="Vector"/> with magnitude one that is parallel to the original <see cref="Vector"/>.</returns>
        public Vector Normalized()
        {
            float magnitude = Magnitude;

            if (magnitude == 0)
                return this;

            return this * (1 / magnitude);
        }

        /// <inheritdoc />
        public override void SyncCPU()
        {
            if (ID == 0)
                return;

            MemoryBuffer buffer = GetBuffer();

            if (buffer != null)
                SyncCPU(buffer);
        }

        /// <inheritdoc />
        public override void SyncCPU(MemoryBuffer buffer)
        {
            buffer.AsArrayView<float>(0, Length).CopyToCPU(_values);
        }
    }
}