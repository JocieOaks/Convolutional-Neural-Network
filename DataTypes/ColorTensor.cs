using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Diagnostics;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="ColorTensor"/> class represents a 2D array of <see cref="Color"/>s.
    /// </summary>
    [Serializable]
    public class ColorTensor
    {
        [JsonProperty] protected Color[] _tensor;

        /// <summary>
        /// Initializes a new <see cref="ColorTensor"/> with the given dimensions.
        /// </summary>
        /// <param name="width">The width of the <see cref="ColorTensor"/>.</param>
        /// <param name="length">The length of the <see cref="ColorTensor"/>.</param>
        public ColorTensor(int width, int length)
        {
            Width = width;
            Length = length;

            _tensor = new Color[width * length];
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor] protected ColorTensor() { }

        /// <value>The full area of the <see cref="ColorTensor"/>.</value>
        [JsonIgnore] public int Area => _tensor.Length;

        /// <value>The length of the <see cref="ColorTensor"/> when converted into an array of floats.</value>
        [JsonIgnore] public int FloatLength => _tensor.Length * 3;

        /// <value>The y length of the <see cref="ColorTensor"/>.</value>
        [JsonProperty] public int Length { get; private set; }

        /// <value>The x width of the <see cref="ColorTensor"/>.</value>
        [JsonProperty] public int Width { get; private set; }

        /// <summary>
        /// Indexes the <see cref="ColorTensor"/> to retrieve the <see cref="Color"/> at the given coordinates.
        /// </summary>
        /// <param name="x">The x coordinate of the desired <see cref="Color"/>.</param>
        /// <param name="y">The y coordinate of the desired <see cref="Color"/>.</param>
        /// <returns>Returns the <see cref="Color"/> at (<paramref name="x"/>, <paramref name="y"/>).</returns>
        public Color this[int x, int y]
        {
            get => _tensor[y * Width + x];
            set => _tensor[y * Width + x] = value;
        }

        private Color this[int index]
        {
            get => _tensor[index];
            set => _tensor[index] = value;
        }

        /// <summary>
        /// Multiplies a <see cref="Vector"/> and a <see cref="ColorTensor"/> by performing tensor contraction using the <see cref="Vector"/> as 
        /// n x 3 tensor and the <see cref="ColorTensor"/> is an n x m x 3 tensor, resulting in a m x 3 tensor.
        /// </summary>
        /// <param name="vector">The <see cref="Vector"/> of length n.</param>
        /// <param name="tensor">The <see cref="ColorTensor"/> of dimensions n x m.</param>
        /// <returns>Returns a <see cref="ColorVector"/> of length m, where m is the <see cref="ColorTensor.Length"/> of <paramref name="tensor"/>.</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="vector"/> length is not equal to <paramref name="tensor"/> width.</exception>
        public static ColorVector operator *(Vector vector, ColorTensor tensor)
        {
            if (tensor.Width != vector.Length)
                throw new ArgumentException("Matrix and vector are not compatible.");
            ColorVector output = new(tensor.Length);
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
        /// Multiplies a <see cref="ColorTensor"/> of dimensions n x m x 3, by a <see cref="ColorVector"/> of dimensions m x 3,
        /// performing double tensor contraction to get a vector of length n.
        /// </summary>
        /// <param name="matrix">The first tensor of dimensions n x m x 3.</param>
        /// <param name="vector">The <see cref="ColorVector"/>, a  tensor of dimensions m x 3.</param>
        /// <returns>Returns a new <see cref="Vector"/> of length equal to <paramref name="matrix"/> width.</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="matrix"/> <see cref="ColorTensor.Length"/> is not equal to
        /// <paramref name="vector"/>'s length.</exception>
        public static Vector operator *(ColorTensor matrix, ColorVector vector)
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

        public static bool operator ==(ColorTensor v1, ColorTensor v2)
        {
            if (v1 is null)
                return v2 is null;
            if (v2 is null)
                return v1 is null;

            if (v1.Length != v2.Length || v1.Width != v2.Width)
                return false;

            for (int i = 0; i < v1.Area; i++)
            {
                if (v1[i] != v2[i])
                    return false;
            }
            return true;
        }

        public static bool operator !=(ColorTensor v1, ColorTensor v2)
        {
            if (v1 is null)
                return v2 is not null;

            if (v2 is null)
                return v1 is not null;

            if (v1.Length != v2.Length || v1.Width != v2.Width)
                return true;

            for (int i = 0; i < v1.Area; i++)
            {
                if (v1[i] != v2[i])
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Generates a new <see cref="ColorTensor"/> of the given dimensions with randomized values. Used for creating starting noise
        /// for a <see cref="Networks.Generator"/>.
        /// </summary>
        /// <param name="width"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        public static ColorTensor Random(int width, int length, float mean, float stdDev)
        {
            ColorTensor map = new(width, length);
            for (int y = 0; y < length; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    map[x, y] = new Color(Utility.RandomGauss(mean, stdDev), 0, 0);
                }
            }
            return map;
        }

        /// <summary>
        /// Allocates a <see cref="MemoryBuffer1D{T, TStride}"/> on the given <see cref="Accelerator"/> with <see cref="ColorTensor"/>'s values.
        /// </summary>
        /// <param name="accelerator">The <see cref="Accelerator"/> on which the map's data is being allocated.</param>
        /// <returns>Returns the created <see cref="MemoryBuffer1D{T, TStride}"/>.</returns>
        public MemoryBuffer1D<Color, Stride1D.Dense> Allocate(Accelerator accelerator)
        {
            return accelerator.Allocate1D(_tensor);
        }

        /// <summary>
        /// Allocates a <see cref="MemoryBuffer1D{T, TStride}"/> on the given <see cref="Accelerator"/> with <see cref="ColorTensor"/>'s length.
        /// </summary>
        /// <param name="accelerator">The <see cref="Accelerator"/> on which the map's data is being allocated.</param>
        /// <returns>Returns the created <see cref="MemoryBuffer1D{T, TStride}"/>.</returns>
        public MemoryBuffer1D<Color, Stride1D.Dense> AllocateEmpty(Accelerator accelerator)
        {
            return accelerator.Allocate1D<Color>(Area);
        }

        /// <summary>
        /// Allocates a <see cref="MemoryBuffer1D{T, TStride}"/> of floats on the given <see cref="Accelerator"/> with <see cref="ColorTensor"/>'s length.
        /// </summary>
        /// <param name="accelerator">The <see cref="Accelerator"/> on which the map's data is being allocated.</param>
        /// <returns>Returns the created <see cref="MemoryBuffer1D{T, TStride}"/>.</returns>
        public MemoryBuffer1D<float, Stride1D.Dense> AllocateFloat(Accelerator accelerator, bool zero)
        {
            var buffer = accelerator.Allocate1D<float>(FloatLength);
            if (zero)
                buffer.MemSetToZero();
            return buffer;
        }

        /// <summary>
        /// Copies the pixel data from a <see cref="MemoryBuffer1D{T, TStride}"/> of <see cref="Color"/>.
        /// </summary>
        /// <param name="buffer">The <see cref="MemoryBuffer1D{T, TStride}"/> with the source <see cref="Color"/>s.</param>
        public void CopyFromBuffer(MemoryBuffer1D<Color, Stride1D.Dense> buffer)
        {
            buffer.AsArrayView<Color>(0, Area).CopyToCPU(_tensor);
        }

        public void CopyFromBuffer(ArrayView<Color> buffer)
        {
            buffer.SubView(0, Area).CopyToCPU(_tensor);
        }

        /// <summary>
        /// Copies the pixel data to a <see cref="MemoryBuffer1D{T, TStride}"/> of <see cref="Color"/>.
        /// </summary>
        /// <param name="buffer">The <see cref="MemoryBuffer1D{T, TStride}"/> to copy to.</param>
        public void CopyToBuffer(MemoryBuffer1D<Color, Stride1D.Dense> buffer)
        {
            buffer.AsArrayView<Color>(0, Area).CopyFromCPU(_tensor);
        }

        public void CopyToBuffer(ArrayView<Color> buffer)
        {
            buffer.SubView(0, Area).CopyFromCPU(_tensor);
        }

        /// <summary>
        /// Copies the pixel data to a <see cref="MemoryBuffer1D{T, TStride}"/> of <see cref="float"/>.
        /// </summary>
        /// <param name="buffer">The <see cref="MemoryBuffer1D{T, TStride}"/> to copy to.</param>
        public void CopyToBuffer(MemoryBuffer1D<float, Stride1D.Dense> buffer)
        {
            unsafe
            {
                fixed (void* ptr = &_tensor[0])
                {
                    Span<float> span = new(ptr, Area * 3);
                    float[] floats = span.ToArray();
                    buffer.AsArrayView<float>(0, Area * 3).CopyFromCPU(floats);
                }
            }
        }

        /// <summary>
        /// Copies the pixel data from a <see cref="MemoryBuffer1D{T, TStride}"/> of floats.
        /// Because <see cref="Color"/> cannot be summed atomically on an <see cref="ILGPU"/> kernal, every three floats represents a single
        /// <see cref="Color"/> in the gradient. The <see cref="ColorTensor"/> is then treated as a <see cref="Span{T}"/> of floats, instead of
        /// an array of <see cref="Color"/>s, copying to memory.
        /// </summary>
        /// <param name="buffer">The <see cref="MemoryBuffer1D{T, TStride}"/> with the source floats.</param>
        public void CopyFromBuffer(MemoryBuffer1D<float, Stride1D.Dense> buffer)
        {
            unsafe
            {
                fixed (void* ptr = &_tensor[0])
                {
                    Span<float> span = new(ptr, Area * 3);
                    buffer.AsArrayView<float>(0, Area * 3).CopyToCPU(span);
                }
            }
        }
    }
}