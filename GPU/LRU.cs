using ILGPU;
using ILGPU.Runtime;
using System.Collections.Concurrent;


namespace ConvolutionalNeuralNetwork.GPU
{
    //Source https://github.com/MPSQUARK/BAVCL

    /// <summary>
    /// The <see cref="LRU"/> class maintains cached data on the CPU using the least recently used caching system.
    /// </summary>
    internal class LRU
    {
        private readonly long _availableMemory;
        private readonly ConcurrentDictionary<uint, Cache> _caches = new();
        private readonly ConcurrentQueue<uint> _lru = new();
        private uint _currentVecId;
        private int _liveObjectCount;
        private long _memoryUsed;

        /// <summary>
        /// Initializes a new instance of the <see cref="LRU"/> class.
        /// </summary>
        /// <param name="maxMemory">The maximum amount of GPU memory available.</param>
        /// <param name="memoryCap">The fraction of the total memory available to the <see cref="LRU"/>, between 0 and 1 inclusive.</param>
        /// <exception cref="Exception">Thrown if <paramref name="memoryCap"/> is not between 0 and 1.</exception>
        public LRU(long maxMemory, float memoryCap)
        {
            if (memoryCap <= 0f || memoryCap >= 1f)
                throw new Exception($"Memory Cap CANNOT be less than 0 or more than 1. Received {memoryCap}");
            _availableMemory = (long)Math.Round(maxMemory * memoryCap);
        }

        /// <summary>
        /// Thread Safe Memory Used Read
        /// </summary>
        public long MemoryUsed => Interlocked.Read(ref _memoryUsed);

        /// <summary>
        /// Allocates memory on the GPU and sets them to be equal to the values of <paramref name="cacheable"/>.
        /// </summary>
        /// <typeparam name="T">The data type of the allocated entries.</typeparam>
        /// <param name="cacheable">The <see cref="Cacheable"/> being allocated.</param>
        /// <param name="accelerator">The <see cref="Accelerator"/> on which to allocate room.</param>
        /// <returns>Returns a tuple containing the key of the stored <see cref="Cache"/> and the allocated <see cref="MemoryBuffer"/>.</returns>
        public (uint, MemoryBuffer) Allocate<T>(Cacheable<T> cacheable, Accelerator accelerator) where T : unmanaged
        {
            uint id = GenerateId();
            MemoryBuffer1D<T, Stride1D.Dense> buffer;

            lock (this)
            {
                ClearSpace(cacheable.MemorySize);
                UpdateMemoryUsage(cacheable.MemorySize);

                buffer = accelerator.Allocate1D(cacheable.GetValues());
                _caches.TryAdd(id, new Cache(buffer, new WeakReference<Cacheable>(cacheable)));
                _lru.Enqueue(id);
            }

            AddLiveTask();
            return (id, buffer);
        }

        /// <summary>
        /// Allocates memory on the GPU where the initial values are unspecified.
        /// </summary>
        /// <typeparam name="T">The data type of the allocated entries.</typeparam>
        /// <param name="cacheable">The <see cref="Cacheable"/> being allocated.</param>
        /// <param name="length">The size of the buffer to be allocated.</param>
        /// <param name="accelerator">The <see cref="Accelerator"/> on which to allocate room.</param>
        /// <returns>Returns a tuple containing the key of the stored <see cref="Cache"/> and the allocated <see cref="MemoryBuffer"/>.</returns>
        public (uint, MemoryBuffer) AllocateEmpty<T>(Cacheable cacheable, int length, Accelerator accelerator) where T : unmanaged
        {
            uint id = GenerateId();
            MemoryBuffer1D<T, Stride1D.Dense> buffer;

            long memNeeded = Interop.SizeOf<T>() * (long)length;

            lock (this)
            {
                ClearSpace(memNeeded);
                UpdateMemoryUsage(memNeeded);

                buffer = accelerator.Allocate1D<T>(length);
                _caches.TryAdd(id, new Cache(buffer, new WeakReference<Cacheable>(cacheable)));
                _lru.Enqueue(id);
            }
            AddLiveTask();
            return (id, buffer);
        }

        /// <summary>
        /// Ensures there is a enough space on the GPU for allocation by clearing out entries least recently used first, until there is the given memory space required.
        /// </summary>
        /// <param name="memRequired">The number of bytes to make available to new allocations.</param>
        /// <exception cref="Exception">Thrown if the size requested is larger than the total space available to the <see cref="LRU"/></exception>
        public void ClearSpace(long memRequired)
        {
            // Check if the memory required doesn't exceed the Maximum available
            if (memRequired > _availableMemory)
                throw new Exception($"Cannot cache this data onto the GPU, required memory : {memRequired >> 20} MB, max memory available : {_availableMemory >> 20} MB.\n " +
                                    $"Consider splitting/breaking the data into multiple smaller sets OR \n Caching to a GPU with more available memory.");

            // Continue to decache until enough space is made to accomodate the data
            while (memRequired + MemoryUsed > _availableMemory)
            {

                if (_liveObjectCount == 0)
                    throw new Exception(
                        $"GPU states {_liveObjectCount} Live Tasks Running, while requiring {memRequired >> 20} MB which is more than available {_availableMemory - MemoryUsed >> 20} MB. Potential cause: memory leak");

                lock (this)
                {
                    // Get the ID of the last item
                    if (!_lru.TryDequeue(out uint id)) throw new Exception("LRU Empty Cannot Continue DeCaching");

                    // Try Get Reference to and the object of ICacheable
                    if (_caches.TryGetValue(id, out Cache cache))
                    {
                        if (IsICacheableLive(cache, id)) continue;

                        cache.MemoryBuffer.Dispose();
                        UpdateMemoryUsage(-cache.MemoryBuffer.LengthInBytes);
                        SubtractLiveTask();
                        _caches.TryRemove(id, out _);
                    }
                }
            }
        }

        /// <summary>
        /// Accesses the <see cref="MemoryBuffer"/> associated with the given id.
        /// </summary>
        /// <returns>Returns the <see cref="MemoryBuffer"/> or null if no <see cref="MemoryBuffer"/> is found.</returns>
        public MemoryBuffer GetBuffer(uint id)
        {
            lock (_caches)
            {
                if (_caches.TryGetValue(id, out Cache cache))
                    return cache.MemoryBuffer;
            }

            return null;
        }

        /// <summary>
        /// Removes cache with the given ID from the <see cref="LRU"/> and disposes the associated <see cref="MemoryBuffer"/>.
        /// </summary>
        /// <param name="id">The ID of the cache to be removed.</param>
        /// <returns>Returns <paramref name="id"/> if the cached item is currently in use by a live process and cannot be removed, 0 otherwise.</returns>
        public uint RemoveItem(uint id)
        {
            lock (this)
            {
                if (!_caches.TryGetValue(id, out Cache cache)) return 0;

                if (IsICacheableLive(cache, id)) return id;

                cache.MemoryBuffer.Dispose();
                UpdateMemoryUsage(-cache.MemoryBuffer.LengthInBytes);
                SubtractLiveTask();
                _caches.TryRemove(id, out _);
                RemoveFromLRU(id);
            }

            return 0;
        }

        private void AddLiveTask() => Interlocked.Increment(ref _liveObjectCount);

        private uint GenerateId() => Interlocked.Increment(ref _currentVecId);

        private bool IsICacheableLive(Cache cache, uint id)
        {
            // If not present, will say data can be decached
            if (!cache.CachedObjRef.TryGetTarget(out Cacheable cacheable))
                return false;

            // If not live, will sync the data back to the cpu
            if (cacheable.LiveCount == 0)
            {
                cacheable.SyncCPU(cache.MemoryBuffer);
                return false;
            }

            // If ICacheable is live, it will re-add it back to the LRU
            _lru.Enqueue(id);
            return true;
        }

        private void RemoveFromLRU(uint id)
        {
            if (_lru.IsEmpty || !_lru.Contains(id)) return;

            _lru.TryDequeue(out uint dequeuedId);

            if (dequeuedId == id) return;

            // Put back the De-queued Id since it didn't match the Disposed Id
            _lru.Enqueue(dequeuedId);

            // shuffle through LRU to remove the disposed Id
            for (int i = 0; i < _lru.Count; i++)
            {
                _lru.TryDequeue(out dequeuedId);

                // Id matching the one disposed will not be re-enqueued 
                // Order will be preserved
                if (id != dequeuedId)
                {
                    _lru.Enqueue(dequeuedId);
                }
            }
        }

        private void SubtractLiveTask() => Interlocked.Decrement(ref _liveObjectCount);

        private void UpdateMemoryUsage(long size) => Interlocked.Add(ref _memoryUsed, size);
    }
}
