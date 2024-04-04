using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace ConvolutionalNeuralNetwork.GPU
{
    /// <summary>
    /// The <see cref="GPUManager"/> class interfaces between the program and <see cref="ILGPU"/>.
    /// </summary>
    public static class GPUManager
    {
        private static readonly LRU s_lru;

        static GPUManager()
        {
            Context = Context.Create(builder => builder.Cuda().EnableAlgorithms().Profiling());
            Accelerator = Context.CreateCudaAccelerator(0);
            s_lru = new LRU(Accelerator.MemorySize, 0.7f);
            CopyAction = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(CopyKernel);
        }

        /// <value>A CUDA <see cref="ILGPU.Runtime.Accelerator"/> for running GPU kernels.</value>
        public static Accelerator Accelerator { get; }

        /// <value>A CUDA <see cref="ILGPU.Context"/> for running GPU kernels.</value>
        public static Context Context { get; }

        /// <value>An action for running <see cref="CopyKernel"/>.</value>
        public static Action<Index1D, ArrayView<float>, ArrayView<float>> CopyAction { get; }

        /// <summary>
        /// Allocates memory on the GPU and sets them to be equal to the values of <paramref name="cacheable"/>.
        /// </summary>
        /// <typeparam name="T">The data type of the allocated entries.</typeparam>
        /// <param name="cacheable">The <see cref="Cacheable"/> being allocated.</param>
        /// <returns>Returns a tuple containing the key of the stored <see cref="Cache"/> and the allocated <see cref="MemoryBuffer"/>.</returns>
        public static (uint, MemoryBuffer) Allocate<T>(Cacheable<T> cacheable) where T : unmanaged => s_lru.Allocate(cacheable, Accelerator);

        /// <summary>
        /// Allocates memory on the GPU where the initial values are unspecified.
        /// </summary>
        /// <typeparam name="T">The data type of the allocated entries.</typeparam>
        /// <param name="cacheable">The <see cref="Cacheable"/> being allocated.</param>
        /// <param name="length">The size of the buffer to be allocated.</param>
        /// <returns>Returns a tuple containing the key of the stored <see cref="Cache"/> and the allocated <see cref="MemoryBuffer"/>.</returns>
        public static (uint, MemoryBuffer) AllocateEmpty<T>(Cacheable cacheable, int length) where T : unmanaged => s_lru.AllocateEmpty<T>(cacheable, length, Accelerator);

        /// <summary>
        /// Removes cache with the given ID from the <see cref="LRU"/> and disposes the associated <see cref="MemoryBuffer"/>.
        /// </summary>
        /// <param name="id">The ID of the cache to be removed.</param>
        /// <returns>Returns <paramref name="id"/> if the cached item is currently in use by a live process and cannot be removed, 0 otherwise.</returns>
        public static uint RemoveItem(uint id) => s_lru.RemoveItem(id);

        /// <summary>
        /// Accesses the <see cref="MemoryBuffer"/> associated with the given id.
        /// </summary>
        /// <returns>Returns the <see cref="MemoryBuffer"/> or null if no <see cref="MemoryBuffer"/> is found.</returns>
        public static MemoryBuffer TryGetBuffer(uint id) => s_lru.GetBuffer(id);

        /// <summary>
        /// Kernel for copying values from one <see cref="ArrayView{T}"/> to another.
        /// </summary>
        /// <param name="index">The index to iterate over every element in the two <see cref="ArrayView{T}"/>s.</param>
        /// <param name="input">The <see cref="ArrayView{T}"/> being copied from.</param>
        /// <param name="output">The <see cref="ArrayView{T}"/> being copied to.</param>
        private static void CopyKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            output[index] = input[index];
        }
    }
}
