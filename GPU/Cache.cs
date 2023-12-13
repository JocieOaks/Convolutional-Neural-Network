using ILGPU.Runtime;


namespace ConvolutionalNeuralNetwork.GPU
{
    //Source https://github.com/MPSQUARK/BAVCL

    /// <summary>
    /// The <see cref="Cache"/> structure connects the cached data on the GPU with its associated CPU <see cref="Cacheable"/>.
    /// </summary>
    public struct Cache
    {
        /// <value>The <see cref="ILGPU.Runtime.MemoryBuffer"/> containing the cached data on the GPU.</value>
        public MemoryBuffer MemoryBuffer { get; set; }

        /// <value>A <see cref="WeakReference"/> to the associated <see cref="Cacheable"/>.</value>
        public WeakReference<Cacheable> CachedObjRef { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Cache"/> class.
        /// </summary>
        /// <param name="memoryBuffer">The GPU cached <see cref="ILGPU.Runtime.MemoryBuffer"/>.</param>
        /// <param name="cachedObjRef">The <see cref="Cacheable"/> on the CPU.</param>
        public Cache(MemoryBuffer memoryBuffer, WeakReference<Cacheable> cachedObjRef)
        {
            MemoryBuffer = memoryBuffer;
            CachedObjRef = cachedObjRef;
        }
    }
}
