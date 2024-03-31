using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="ByteArray"/> class stores a cacheable array of bytes.
    /// </summary>
    [Serializable]
    public class ByteArray : Cacheable<byte>
    {
        private readonly byte[] _values;

        /// <summary>
        /// Initializes a new <see cref="ByteArray"/> using an array of bytes.
        /// </summary>
        /// <param name="values">The values of the <see cref="ByteArray"/>.</param>
        public ByteArray(byte[] values)
        {
            _values = values;
        }

        /// <summary>
        /// Initializes a new empty <see cref="ByteArray"/> of a given length.
        /// </summary>
        /// <param name="length">The number of bytes contained in the <see cref="ByteArray"/>.</param>
        public ByteArray(int length)
        {
            _values = new byte[length];
        }

        /// <value>The length of the <see cref="ByteArray"/>.</value>
        public int Length => _values.Length;

        /// <inheritdoc/>
        public override long MemorySize => Length;

        /// <summary>
        /// Indexes the <see cref="ByteArray"/> retrieving the value at the desired index.
        /// </summary>
        public byte this[int index]
        {
            get => _values[index];
            set => _values[index] = value;
        }

        /// <summary>
        /// Copies the <see cref="ByteArray"/> to an <see cref="ArrayView{T}"/>
        /// </summary>
        public void CopyToView(ArrayView<byte> view)
        {
            view.SubView(0, Length).CopyFromCPU(_values);
        }

        /// <summary>
        /// Copies the <see cref="ByteArray"/> to the cached <see cref="ArrayView{T}"/>
        /// allocating it if it is not cached.
        /// </summary>
        public void CopyToView()
        {
            GetArrayView().CopyFromCPU(_values);
            Release();
        }

        /// <summary>
        /// Gets the <see cref="ArrayView{T}"/> for the cached <see cref="ByteArray"/> or allocates it if the <see cref="ByteArray"/> is decached.
        /// </summary>
        /// <returns>Returns an <see cref="ArrayView{T}"/>.</returns>
        public ArrayView<byte> GetArrayView()
        {
            Live();
            MemoryBuffer buffer = GetBuffer();
            if (buffer == null)
            {
                (ID, buffer) = GPUManager.Allocate(this);
            }
            return new ArrayView<byte>(buffer, 0, Length);
        }

        /// <inheritdoc />
        public override byte[] GetValues()
        {
            return _values;
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
            buffer.AsArrayView<byte>(0, Length).CopyToCPU(_values);
        }
    }
}