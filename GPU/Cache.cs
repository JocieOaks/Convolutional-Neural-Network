using ILGPU.Runtime;
using System;


namespace ConvolutionalNeuralNetwork.GPU
{
    //Code from https://github.com/MPSQUARK/BAVCL
    public struct Cache
    {
        public MemoryBuffer MemoryBuffer;
        public WeakReference<Cacheable> CachedObjRef;

        public Cache(MemoryBuffer memoryBuffer, WeakReference<Cacheable> cachedObjRef)
        {
            MemoryBuffer = memoryBuffer;
            CachedObjRef = cachedObjRef;
        }
    }
}
