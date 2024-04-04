using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.GPU
{
    //Source https://github.com/MPSQUARK/BAVCL

    /// <summary>
    /// The <see cref="Cacheable"/> class is an abstract class for data structures that are cached on the GPU.
    /// </summary>
    public abstract class Cacheable
    {
        /// <value>The identifier for <see cref="LRU"/>.</value>
        [JsonIgnore] public uint ID { get; set; }

        /// <value>The number of live processes currently using the cached data.</value>
        [JsonIgnore] public uint LiveCount { get; private set; }

        /// <value>The size in bytes of the cached memory taken up by this <see cref="Cacheable"/>.</value>
        public abstract long MemorySize { get; }

        /// <summary>
        /// Removes the cached data from the GPU and stores it on the CPU. Fails if there is a live process currently using the cached data.
        /// </summary>
        public void DeCache()
        {
            // If the tensor is not cached - it's technically already decached
            if (ID == 0)
                return;

            // If the tensor is live - Fail
            if (LiveCount != 0)
                return;

            // Else Decache
            SyncCPU();
            ID = GPUManager.RemoveItem(ID);
        }

        /// <summary>
        /// Called when a process that uses the cached data goes live.
        /// </summary>
        public void Live()
        {
            LiveCount ++;
            if( LiveCount > 200 )
            {
                Console.WriteLine("Live Count exceeding limit.");
            }
        }

        /// <summary>
        /// Called when a process that uses the cached data completes.
        /// </summary>
        public void Release()
        {
            LiveCount --;
            if (LiveCount > 200)
            {
                Console.WriteLine("Live Count exceeding limit.");
            }
        }

        /// <summary>
        /// Syncs the data on the CPU with the data stored on the GPU.
        /// </summary>
        public abstract void SyncCPU();

        /// <summary>
        /// Syncs the data on the CPU with the data stored on the GPU using the specified <see cref="MemoryBuffer"/>.
        /// </summary>
        /// <param name="buffer">The <see cref="MemoryBuffer"/> containing the cached data.</param>
        public abstract void SyncCPU(MemoryBuffer buffer);

        /// <summary>
        /// Get the <see cref="MemoryBuffer"/> from the <see cref="GPUManager"/>.
        /// </summary>
        /// <returns>Returns the <see cref="MemoryBuffer"/> containing the cached data. Returns null if data is decached.</returns>
        protected MemoryBuffer GetBuffer()
        {
            return GPUManager.TryGetBuffer(ID);
        }
    }

    /// <summary>
    /// The <see cref="Cacheable{T}"/> class is an extension of <see cref="Cacheable"/> where data entries are structured as an array of type <typeparamref name="T"/>.
    /// </summary>
    /// <typeparam name="T">The data type of the <see cref="Cacheable"/>'s data entries.</typeparam>
    public abstract class Cacheable<T> : Cacheable where T : unmanaged
    {
        /// <summary>
        /// Access the list of data entries stored as an array of <typeparamref name="T"/>.
        /// </summary>
        /// <returns>Returns an array of <typeparamref name="T"/>.</returns>
        public abstract T[] GetValues();
    }
}
