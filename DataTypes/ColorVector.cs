using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="ColorVector"/> class stores an array of <see cref="Color"/>s for performing vector mathematics.
    /// Technically because <see cref="Color"/> is itself a vector, <see cref="ColorVector"/> is more accurately a rank two tensor. However,
    /// for simplification <see cref="Color"/> is generally treated as a scaler value.
    /// </summary>
    [Serializable]
    public class ColorVector : Cacheable<Color>
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

        public ColorVector(Color[] values)
        {
            _values = values;
        }

        public int FloatLength => 3 * Length;

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

        public override long MemorySize => Length * 12;

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

        public override Color[] GetValues()
        {
            return _values;
        }

        public ArrayView<T> GetArrayViewEmpty<T>() where T : unmanaged
        {
            IncrementLiveCount();
            MemoryBuffer buffer = GetBuffer();
            if (buffer == null)
            {
                (ID, buffer) = GPUManager.AllocateEmpty<Color>(this, Length);
            }
            int bytes = Interop.SizeOf<T>();
            return new ArrayView<T>(buffer, 0, 12 * Length / bytes);
        }

        public ArrayView<T> GetArrayView<T>() where T : unmanaged
        {
            IncrementLiveCount();
            MemoryBuffer buffer = GetBuffer();
            if (buffer == null)
            {
                (ID, buffer) = GPUManager.Allocate(this);
            }
            int bytes = Interop.SizeOf<T>();
            return new ArrayView<T>(buffer, 0, 12 * Length / bytes);
        }

        public ArrayView<T> GetArrayViewZeroed<T>() where T : unmanaged
        {
            ArrayView<T> arrayView = GetArrayView<T>();
            arrayView.MemSetToZero();
            return arrayView;
        }

        public override void SyncCPU(ArrayView<Color> arrayView)
        {
            arrayView.SubView(0, Length).CopyToCPU(_values);
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
            if (ID == 0)
                return;

            MemoryBuffer buffer = GetBuffer();

            if (buffer != null)
                SyncCPU(buffer);
        }

        public override void SyncCPU(MemoryBuffer buffer)
        {
            buffer.AsArrayView<Color>(0, Length).CopyToCPU(_values);
        }

        public void UpdateIfAllocated()
        {
            if (ID == 0)
                return;

            MemoryBuffer buffer = GetBuffer();
            if (buffer == null)
                return;

            buffer.AsArrayView<Color>(0, Length).CopyFromCPU(_values);
        }
    }
}