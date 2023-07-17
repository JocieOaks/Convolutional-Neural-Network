using ConvolutionalNeuralNetwork.GPU;
using ConvolutionalNeuralNetwork.Layers;
using ILGPU;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="IOBuffers"/> class contains a set of <see cref="MemoryBuffer"/>s for use with an <see cref="ILGPU"/> kernel.
    /// Because the output of one <see cref="Layer"/> is the input of another layer, two set of these can be used such that the device inputs of one
    /// is the device outputs of another. This saves a significant amount of time instead of allocating and then deallocating <see cref="MemoryBuffer"/>s.
    /// </summary>
    public class IOBuffers
    {
        ArrayView<float>[,] _floats1;
        ArrayView<float>[,] _floats2;


        readonly List<(int dimensions, int area)> _outputAllocationPairs = new();
        public ArrayView<float>[,] InGradientsFloat => _floats1;
        public ArrayView<float>[,] InputsFloat => _floats2;
        public ArrayView<float>[,] OutGradientsFloat => _floats2;
        public ArrayView<float>[,] OutputsFloat => _floats1;

        public ArrayView<float> FinalOutput(int batchIndex) => _floats1[0, batchIndex];
        public ArrayView<float> FirstGradient(int batchIndex) => _floats1[0, batchIndex];

        /// <summary>
        /// Sets to <see cref="IOBuffers"/> to be reflections of eachother. Aka, the input of one is the output of the other.
        /// </summary>
        /// <param name="buffers1">The first <see cref="IOBuffers"/>.</param>
        /// <param name="buffers2">The second <see cref="IOBuffers"/>.</param>
        public static void SetCompliment(IOBuffers buffers1, IOBuffers buffers2)
        {
            (buffers1._floats2, buffers2._floats2) = (buffers2._floats1, buffers1._floats1);
        }

        /// <summary>
        /// Allocate buffers for the inputs, outputs and gradients.
        /// </summary>
        /// <param name="batchSize">The number of elements in a single batch.</param>
        public void Allocate(int batchSize)
        {
            int dimensions = _outputAllocationPairs.MaxBy(x => x.dimensions).dimensions;
            _floats1 = new ArrayView<float>[dimensions, batchSize];
            long memoryUsage = 0;

            for (int i = 0; i < dimensions; i++)
            {
                int area = _outputAllocationPairs.Where(x => x.dimensions > i).MaxBy(x => x.area).area;
                for (int j = 0; j < batchSize; j++)
                {
                    var buffer = GPUManager.Accelerator.Allocate1D<float>(area);
                    _floats1[i, j] = new ArrayView<float>(buffer, 0, area);
                    memoryUsage += area * 4;
                }
            }

            GPUManager.AddExternalMemoryUsage(memoryUsage);
        }

        /// <summary>
        /// Adds a new pair detailing the maximum number of dimensions needed for the buffers, and the maximum space needed to be allocated for those dimensions.
        /// </summary>
        /// <param name="dimensions"></param>
        /// <param name="area"></param>
        public void OutputDimensionArea(int dimensions, int area)
        {
            _outputAllocationPairs.Add((dimensions + 1, area));
        }
    }
}