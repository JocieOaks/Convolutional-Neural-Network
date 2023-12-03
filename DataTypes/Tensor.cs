using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="Tensor"/> class represents a 2D array of floats.
    /// </summary>
    [Serializable]
    public class Tensor : Cacheable<float>
    {
        /// <value>The values of the <see cref="Tensor"/> as a single dimensional array.</value>
        [JsonProperty] protected float[] Values;

        /// <summary>
        /// Initializes a new <see cref="Tensor"/> with the given dimensions.
        /// </summary>
        /// <param name="width">The width of the <see cref="Tensor"/>.</param>
        /// <param name="length">The length of the <see cref="Tensor"/>.</param>
        public Tensor(int width, int length)
        {
            Width = width;
            Length = length;

            Values = new float[width * length];
        }

        /// <summary>
        /// Initializes a new <see cref="Tensor"/> with the given <see cref="Shape"/>.
        /// </summary>
        /// <param name="shape">The <see cref="Shape"/> of the new <see cref="Tensor"/>. Ignores dimension.</param>
        public Tensor(Shape shape)
        {
            Width = shape.Width;
            Length = shape.Length;

            Values = new float[shape.Area];
        }

        /// <summary>
        /// Initializes a new <see cref="Tensor"/> with the given dimensions, initialized with a given value.
        /// </summary>
        /// <param name="width">The width of the <see cref="Tensor"/>.</param>
        /// <param name="length">The length of the <see cref="Tensor"/>.</param>
        /// <param name="value">The value to set every entry in the <see cref="Tensor"/> to.</param>
        public Tensor(int width, int length, float value)
        {
            for (int i = 0; i < Area; i++)
            {
                Values[i] = value;
            }
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor] protected Tensor() { }

        /// <value>The full area of the <see cref="Tensor"/>.</value>
        [JsonIgnore] public int Area => Values.Length;

        /// <value>The y length of the <see cref="Tensor"/>.</value>
        [JsonProperty] public int Length { get; private set; }

        /// <inheritdoc/>
        public override long MemorySize => Area * 12;

        /// <value>The x width of the <see cref="Tensor"/>.</value>
        [JsonProperty] public int Width { get; private set; }
        /// <summary>
        /// Indexes the <see cref="Tensor"/> to retrieve the value at the given coordinates.
        /// </summary>
        /// <param name="x">The x coordinate of the desired value.</param>
        /// <param name="y">The y coordinate of the desired value.</param>
        /// <returns>Returns the value at (<paramref name="x"/>, <paramref name="y"/>).</returns>
        public float this[int x, int y]
        {
            get => Values[y * Width + x];
            set => Values[y * Width + x] = value;
        }

        /// <summary>
        /// Multiplies a <see cref="Vector"/> and a <see cref="Tensor"/> by performing tensor contraction using the <see cref="Vector"/> as 
        /// n tensor and the <see cref="Tensor"/> is an n x m tensor, resulting in a m tensor.
        /// </summary>
        /// <param name="vector">The <see cref="Vector"/> of length n.</param>
        /// <param name="tensor">The <see cref="Tensor"/> of dimensions n x m.</param>
        /// <returns>Returns a <see cref="Vector"/> of length m, where m is the <see cref="Tensor.Length"/> of <paramref name="tensor"/>.</returns>
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
        /// Copies the <see cref="Tensor"/> to an <see cref="ArrayView{T}"/>
        /// </summary>
        /// <param name="view">The <see cref="ArrayView{T}"/> being copied to.</param>
        public void CopyToView(ArrayView<float> view)
        {
            view.SubView(0, Area).CopyFromCPU(Values);
        }

        /// <inheritdoc/>
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

        /// <summary>
        /// Gets the <see cref="ArrayView{T}"/> for the cached <see cref="Tensor"/> or allocates it if the <see cref="Tensor"/> is decached.
        /// </summary>
        /// <returns></returns>
        public ArrayView<float> GetArrayView()
        {
            IncrementLiveCount();
            MemoryBuffer buffer = GetBuffer();
            if (buffer == null)
            {
                (ID, buffer) = GPUManager.Allocate(this);
            }
            int bytes = Interop.SizeOf<float>();
            return new ArrayView<float>(buffer, 0, 12 * Area / bytes);
        }

        /// <summary>
        /// Gives all the values of the <see cref="Tensor"/>.
        /// </summary>
        /// <returns>Returns the <see cref="Tensor"/>'s values as a single dimensional array of floats.</returns>
        public override float[] GetValues()
        {
            return Values;
        }

        /// <inheritdoc />
        public override void SyncCPU()
        {
            if(ID != 0)
                SyncCPU(GetBuffer());
        }

        /// <inheritdoc />
        public override void SyncCPU(MemoryBuffer buffer)
        {
            buffer.AsArrayView<float>(0, Area).CopyToCPU(Values);
        }

        /// <inheritdoc />
        public override void SyncCPU(ArrayView<float> arrayView)
        {
            arrayView.SubView(0, Area).CopyToCPU(Values);
        }
    }
}