using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;

namespace ConvolutionalNeuralNetwork.GPU
{
    public static class GPUManager
    {
        private const bool DEBUGCPU = false;
        private static readonly LRU _lru;
        static GPUManager()
        {
            if(DEBUGCPU)
            {
                Context = Context.Create(builder => builder.CPU().EnableAlgorithms().Profiling());
                Accelerator = Context.CreateCPUAccelerator(0);
            }
            else
            {
                Context = Context.Create(builder => builder.Cuda().EnableAlgorithms().Profiling());
                Accelerator = Context.CreateCudaAccelerator(0); 
            }
            _lru = new LRU(Accelerator.MemorySize, 0.8f);
            CopyAction = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(CopyKernel);
            AddAction = Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, int>(AddKernel);
        }

        /// <value>A Cuda <see cref="global::ILGPU.Runtime.Accelerator"/> for running <see cref="global::ILGPU"/> kernels.</value>
        public static Accelerator Accelerator { get; }

        public static Context Context { get; }
        /// <value>An action for running the cuda kernel <see cref="CopyKernel(Index1D, ArrayView{Color}, ArrayView{Color})"/>.</value>
        public static Action<Index1D, ArrayView<float>, ArrayView<float>> CopyAction { get; }

        public static Action<Index3D, ArrayView<float>, ArrayView<float>, int> AddAction { get; }



        public static (uint, MemoryBuffer) Allocate<T>(Cacheable<T> cacheable) where T : unmanaged => _lru.Allocate(cacheable, Accelerator);

        public static (uint, MemoryBuffer) Allocate<T>(Cacheable cacheable, T[] values) where T : unmanaged => _lru.Allocate(cacheable, values, Accelerator);

        public static (uint, MemoryBuffer) AllocateEmpty<T>(Cacheable cacheable, int length) where T : unmanaged => _lru.AllocateEmpty<T>(cacheable, length, Accelerator);

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

        /// <summary>
        /// An <see cref="ILGPU"/> kernel that adds the values from one <see cref="ArrayView{T}"/> of floats, to another. Each array is broken up into
        /// subarrays of equal size, then every subarray from <paramref name="addition"/> is added to every subarray of <paramref name="value"/>.
        /// Typically one of the two arrays only contains one subarray, with a single <paramref name="addition"/> subarray being added to every <paramref name="value"/> subarray
        /// or multiple <paramref name="addition"/> subarrays added to a single <paramref name="value"/> subarray.
        /// </summary>
        /// <param name="index">The index of the arrays to sum.
        /// X iterates over a single subarray.
        /// Y iterates over the subarrays of <paramref name="value"/>.
        /// Z iterates over the subarrays of <paramref name="addition"/>.</param>
        /// <param name="value"> The array of floats to which <paramref name="addition"/> is being added.</param>
        /// <param name="addition">The array of floats being added to <paramref name="value"/>.</param>
        /// <param name="subarrayArea">The size of each subarray.</param>
        private static void AddKernel(Index3D index, ArrayView<float> value, ArrayView<float> addition, int subarrayArea)
        {
            Atomic.Add( ref value[index.Y * subarrayArea + index.X], addition[index.Z * subarrayArea + index.X]);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="index">
        /// X iterates over batchSize
        /// Y iterates over dimension
        /// Z iterates over the area of dimension
        /// </param>
        /// <param name="value"></param>
        /// <param name="bias"></param>


        public static uint GCItem(uint Id) => _lru.GCItem(Id);

        public static string GetMemUsage() => _lru.MemoryUsed.ToString();

        public static MemoryBuffer TryGetBuffer(uint Id) => _lru.GetBuffer(Id);
        public static (uint, MemoryBuffer) UpdateBuffer<T>(Cacheable cacheable, T[] values) where T : unmanaged => _lru.UpdateBuffer(cacheable, values, Accelerator);
        public static (uint, MemoryBuffer) UpdateBuffer<T>(Cacheable<T> cacheable) where T : unmanaged => _lru.UpdateBuffer(cacheable, Accelerator);

        public static void AddExternalMemoryUsage(long size) => _lru.AddExternalMemoryUsage(size);
    }
}
