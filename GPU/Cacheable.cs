using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.GPU
{
    //Code from https://github.com/MPSQUARK/BAVCL
    public abstract class Cacheable
    {
        [JsonIgnore] public uint LiveCount { get; private set; }
        [JsonIgnore] public uint ID { get; set; }
        public abstract long MemorySize { get; }

        public abstract void DeCache();

        public void IncrementLiveCount(uint count = 1)
        {
            LiveCount += count;
            if( LiveCount > 200 )
            {
                Console.WriteLine("Live Count exceeding limit.");
            }
        }

        public void DecrementLiveCount(uint count = 1)
        {
            LiveCount -= count;
            if (LiveCount > 200)
            {
                Console.WriteLine("Live Count exceeding limit.");
            }
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
