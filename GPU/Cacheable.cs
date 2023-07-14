using ILGPU;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.GPU
{
    //Code from https://github.com/MPSQUARK/BAVCL
    public abstract class Cacheable
    {
        public uint LiveCount { get; private set; }
        public uint ID { get; set; }
        public abstract long MemorySize { get; }

        public abstract void DeCache();

        public void IncrementLiveCount()
        {
            LiveCount++;
            if (LiveCount > 100)
                throw new Exception();
        }

        public void DecrementLiveCount()
        {
            LiveCount--;
            if (LiveCount > 100)
                throw new Exception();
        }

        protected MemoryBuffer GetBuffer()
        {
            return GPUManager.TryGetBuffer(ID);
        }

        public abstract void SyncCPU();

        public abstract void SyncCPU(MemoryBuffer buffer);

    }

    public abstract class Cacheable<T> : Cacheable where T : unmanaged
    {
        public abstract T[] GetValues();

        public abstract void SyncCPU(ArrayView<T> arrayView);
    }
}
