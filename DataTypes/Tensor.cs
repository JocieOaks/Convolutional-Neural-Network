using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.IR;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System;
using System.Diagnostics;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="Tensor"/> class represents a 2D array of <see cref="Color"/>s.
    /// </summary>
    [Serializable]
    public class Tensor : Cacheable<float>
    {
        [JsonProperty] protected float[] _tensor;

        /// <summary>
        /// Initializes a new <see cref="Tensor"/> with the given dimensions.
        /// </summary>
        /// <param name="width">The width of the <see cref="Tensor"/>.</param>
        /// <param name="length">The length of the <see cref="Tensor"/>.</param>
        public Tensor(int width, int length)
        {
            Width = width;
            Length = length;

            _tensor = new float[width * length];
        }

        public Tensor(Shape shape)
        {
            Width = shape.Width;
            Length = shape.Length;

            _tensor = new float[shape.Area];
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor] protected Tensor() { }

        /// <value>The full area of the <see cref="Tensor"/>.</value>
        [JsonIgnore] public int Area => _tensor.Length;

        /// <value>The length of the <see cref="Tensor"/> when converted into an array of floats.</value>
        [JsonIgnore] public int FloatLength => _tensor.Length * 3;

        /// <value>The y length of the <see cref="Tensor"/>.</value>
        [JsonProperty] public int Length { get; private set; }

        /// <value>The x width of the <see cref="Tensor"/>.</value>
        [JsonProperty] public int Width { get; private set; }

        public override long MemorySize => Area * 12;

        /// <summary>
        /// Indexes the <see cref="Tensor"/> to retrieve the <see cref="Color"/> at the given coordinates.
        /// </summary>
        /// <param name="x">The x coordinate of the desired <see cref="Color"/>.</param>
        /// <param name="y">The y coordinate of the desired <see cref="Color"/>.</param>
        /// <returns>Returns the <see cref="Color"/> at (<paramref name="x"/>, <paramref name="y"/>).</returns>
        public float this[int x, int y]
        {
            get => _tensor[y * Width + x];
            set => _tensor[y * Width + x] = value;
        }

        private float this[int index]
        {
            get => _tensor[index];
            set => _tensor[index] = value;
        }

        /// <summary>
        /// Multiplies a <see cref="Vector"/> and a <see cref="Tensor"/> by performing tensor contraction using the <see cref="Vector"/> as 
        /// n x 3 tensor and the <see cref="Tensor"/> is an n x m x 3 tensor, resulting in a m x 3 tensor.
        /// </summary>
        /// <param name="vector">The <see cref="Vector"/> of length n.</param>
        /// <param name="tensor">The <see cref="Tensor"/> of dimensions n x m.</param>
        /// <returns>Returns a <see cref="ColorVector"/> of length m, where m is the <see cref="Tensor.Length"/> of <paramref name="tensor"/>.</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="vector"/> length is not equal to <paramref name="tensor"/> width.</exception>
        public static Vector operator *(Vector vector, Tensor tensor)
        {
            if (tensor.Width != vector.Length)
                throw new ArgumentException("Matrix and vector are not compatible.");
            Vector output = new(tensor.Length);
            for (int i = 0; i < tensor.Width; i++)
            {
                for (int j = 0; j < tensor.Length; j++)
                {
                    output[j] += tensor[i, j] * vector[i];
                }
            }

            return output;
        }

        /// <summary>
        /// Multiplies a <see cref="Tensor"/> of dimensions n x m x 3, by a <see cref="ColorVector"/> of dimensions m x 3,
        /// performing double tensor contraction to get a vector of length n.
        /// </summary>
        /// <param name="matrix">The first tensor of dimensions n x m x 3.</param>
        /// <param name="vector">The <see cref="ColorVector"/>, a  tensor of dimensions m x 3.</param>
        /// <returns>Returns a new <see cref="Vector"/> of length equal to <paramref name="matrix"/> width.</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="matrix"/> <see cref="Tensor.Length"/> is not equal to
        /// <paramref name="vector"/>'s length.</exception>
        /*public static Vector operator *(ColorTensor matrix, ColorVector vector)
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
        }*/

        /// <summary>
        /// Generates a new <see cref="ColorTensor"/> of the given dimensions with randomized values. Used for creating starting noise
        /// for a <see cref="Networks.Generator"/>.
        /// </summary>
        /// <param name="width"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        public static Tensor Random(int width, int length, float mean, float stdDev)
        {
            Tensor tensor = new(width, length);
            for (int y = 0; y < length; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    tensor[x, y] = Utility.RandomGauss(mean, stdDev);
                }
            }
            return tensor;
        }

        public void Randomize(float mean, float stdDev)
        {
            for(int i = 0; i < Area; i++)
            {
                _tensor[i] = Utility.RandomGauss(0, stdDev);
            }
        }

        public void CopyToBuffer(ArrayView<float> buffer)
        {
            buffer.SubView(0, Area).CopyFromCPU(_tensor);
        }

        public override float[] GetValues()
        {
            return _tensor;
        }

        public ArrayView<T> GetArrayViewEmpty<T>() where T : unmanaged
        {
            IncrementLiveCount();
            MemoryBuffer buffer = GetBuffer();
            if (buffer == null)
            {
                (ID, buffer) = GPUManager.AllocateEmpty<float>(this, Area);
            }
            int bytes = Interop.SizeOf<T>();
            return new ArrayView<T>(buffer, 0, 12 * Area / bytes);
        }

        public ArrayView<T> GetArrayView<T>() where T: unmanaged
        {
            IncrementLiveCount();
            MemoryBuffer buffer = GetBuffer();
            if(buffer == null)
            {
                (ID, buffer) = GPUManager.Allocate<float>(this);
            }
            int bytes = Interop.SizeOf<T>();
            return new ArrayView<T>(buffer, 0, 12 * Area / bytes);
        }

        public ArrayView<T> GetArrayViewZeroed<T>() where T: unmanaged
        {
            ArrayView<T> arrayView = GetArrayView<T>();
            arrayView.MemSetToZero();
            return arrayView;
        }

        public override void DeCache()
        {
            // If the tensor is not cached - it's technically already decached
            if (ID == 0) 
                return;

            // If the tensor is live - Fail
            if (LiveCount != 0) 
                return;

            // Else Decache
            SyncCPU();
            ID = GPUManager.GCItem(ID);
        }

        public override void SyncCPU()
        {
            if(ID != 0)
                SyncCPU(GetBuffer());
        }

        public override void SyncCPU(MemoryBuffer buffer)
        {
            buffer.AsArrayView<float>(0, Area).CopyToCPU(_tensor);
        }

        public override void SyncCPU(ArrayView<float> arrayView)
        {
            arrayView.SubView(0, Area).CopyToCPU(_tensor);
        }
    }
}