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
        /// Initializes a new <see cref="Tensor"/> with the given <see cref="DataTypes.Shape"/>.
        /// </summary>
        /// <param name="shape">The <see cref="Shape"/> of the new <see cref="Tensor"/>.</param>
        public Tensor(Shape shape)
        {
            Shape = shape;

            Values = new float[shape.Volume];
        }

        /// <summary>
        /// Initializes a new <see cref="Tensor"/> with the given dimensions, initialized with a given value.
        /// </summary>
        /// /// <param name="shape">The <see cref="DataTypes.Shape"/> of the new <see cref="Tensor"/>.</param>
        /// <param name="value">The value to set every entry in the <see cref="Tensor"/> to.</param>
        public Tensor(Shape shape, float value) : this(shape)
        {
            for (int i = 0; i < Volume; i++)
            {
                Values[i] = value;
            }
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor] protected Tensor() { }

        /// <value>The full area of the <see cref="Tensor"/>.</value>
        [JsonIgnore] public int Area => Shape.Area;

        /// <value>The number of dimensions of the <see cref="Tensor"/>.</value>
        [JsonIgnore] public int Dimensions => Shape.Dimensions;

        /// <value>The total size of the tensor.</value>
        [JsonIgnore] public int Volume => Values.Length;

        /// <value>The y length of the <see cref="Tensor"/>.</value>
        [JsonIgnore] public int Length => Shape.Length;

        /// <value>The <see cref="DataTypes.Shape"/> describing the <see cref="Tensor"/>.</value>
        [JsonProperty] public Shape Shape { get; private set; }

        /// <inheritdoc/>
        [JsonIgnore] public override long MemorySize => Volume * 12;

        /// <value>The x width of the <see cref="Tensor"/>.</value>
        [JsonIgnore] public int Width => Shape.Width;

        /// <summary>
        /// Indexes the <see cref="Tensor"/> to retrieve the value at the given coordinates.
        /// </summary>
        /// <param name="x">The x coordinate of the desired value.</param>
        /// <param name="y">The y coordinate of the desired value.</param>
        /// <param name="dimension">The dimension of the desired value.</param>
        /// <returns>Returns the value at (<paramref name="x"/>, <paramref name="y"/>, <param name="dimension"/>).</returns>
        public float this[int x, int y, int dimension]
        {
            get => Values[dimension * Area + y * Width + x];
            set => Values[dimension * Area + y * Width + x] = value;
        }

        /// <summary>
        /// Copies the <see cref="Tensor"/> to an <see cref="ArrayView{T}"/>
        /// </summary>
        /// <param name="view">The <see cref="ArrayView{T}"/> being copied to.</param>
        public void CopyToView(ArrayView<float> view)
        {
            view.SubView(0, Volume).CopyFromCPU(Values);
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
            return new ArrayView<float>(buffer, 0, 12 * Volume / bytes);
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
            buffer.AsArrayView<float>(0, Volume).CopyToCPU(Values);
        }

        /// <inheritdoc />
        public override void SyncCPU(ArrayView<float> arrayView)
        {
            arrayView.SubView(0, Volume).CopyToCPU(Values);
        }
    }
}