using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.GPU
{
    public static class GPUManager
    {

        public static Context Context { get; }

        /// <value>A Cuda <see cref="global::ILGPU.Runtime.Accelerator"/> for running <see cref="global::ILGPU"/> kernals.</value>
        public static Accelerator Accelerator { get; }

        private static readonly LRU _lru;

        /// <value>An action for running the cuda kernal <see cref="CopyKernal(Index1D, ArrayView{Color}, ArrayView{Color})"/>.</value>
        public static Action<Index1D, ArrayView<Color>, ArrayView<Color>> CopyAction { get; }

        /// <summary>
        /// Kernal for copying values from one <see cref="ArrayView{T}"/> to another.
        /// </summary>
        /// <param name="index">The index to iterate over every element in the two <see cref="ArrayView{T}"/>s.</param>
        /// <param name="input">The <see cref="ArrayView{T}"/> being copied from.</param>
        /// <param name="output">The <see cref="ArrayView{T}"/> being copied to.</param>
        public static void CopyKernal(Index1D index, ArrayView<Color> input, ArrayView<Color> output)
        {
            output[index] = input[index];
        }

        static GPUManager()
        {
            Context = Context.Create(builder => builder.Cuda());
            Accelerator = Context.CreateCudaAccelerator(0);
            _lru = new LRU(Accelerator.MemorySize, 0.5f);
            CopyAction = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Color>, ArrayView<Color>>(CopyKernal);
        }

        public static (uint, MemoryBuffer) Allocate<T>(Cacheable<T> cacheable) where T : unmanaged => _lru.Allocate(cacheable, Accelerator);
        public static (uint, MemoryBuffer) Allocate<T>(Cacheable cacheable, T[] values) where T : unmanaged => _lru.Allocate(cacheable, values, Accelerator);
        public static (uint, MemoryBuffer) AllocateEmpty<T>(Cacheable cacheable, int length) where T : unmanaged => _lru.AllocateEmpty<T>(cacheable, length, Accelerator);
        public static MemoryBuffer TryGetBuffer(uint Id) => _lru.GetBuffer(Id);
        public static (uint, MemoryBuffer) UpdateBuffer<T>(Cacheable cacheable, T[] values) where T : unmanaged => _lru.UpdateBuffer(cacheable, values, Accelerator);
        public static (uint, MemoryBuffer) UpdateBuffer<T>(Cacheable<T> cacheable) where T : unmanaged => _lru.UpdateBuffer(cacheable, Accelerator);
        public static uint GCItem(uint Id) => _lru.GCItem(Id);

        public static string GetMemUsage() => _lru.MemoryUsed.ToString();
    }
}
