using ConvolutionalNeuralNetwork.Layers;
using ILGPU;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="IOBuffers"/> class contains a set of <see cref="MemoryBuffer"/>s for use with an <see cref="ILGPU"/> kernal.
    /// Because the output of one <see cref="Layer"/> is the input of another layer, two set of these can be used such that the device inputs of one
    /// is the device outputs of another. This saves a significant amount of time instead of allocating and then deallocating <see cref="MemoryBuffer"/>s.
    /// </summary>
    public class IOBuffers
    {

        ArrayView<Color>[,] _colors1;
        ArrayView<Color>[,] _colors2;

        ArrayView<float>[,] _floats1;
        ArrayView<float>[,] _floats2;


        readonly List<(int dimensions, int area)> _outputAllocationPairs = new();
        public ArrayView<Color>[,] InGradientsColor => _colors1;
        public ArrayView<float>[,] InGradientsFloat => _floats1;
        public ArrayView<Color>[,] InputsColor => _colors2;
        public ArrayView<float>[,] InputsFloat => _floats2;
        public ArrayView<Color>[,] OutGradientsColor => _colors2;
        public ArrayView<float>[,] OutGradientsFloat => _floats2;
        public ArrayView<Color>[,] OutputsColor => _colors1;
        public ArrayView<float>[,] OutputsFloat => _floats1;
        /// <summary>
        /// Sets to <see cref="IOBuffers"/> to be reflections of eachother. Aka, the input of one is the output of the other.
        /// </summary>
        /// <param name="buffers1">The first <see cref="IOBuffers"/>.</param>
        /// <param name="buffers2">The second <see cref="IOBuffers"/>.</param>
        public static void SetCompliment(IOBuffers buffers1, IOBuffers buffers2)
        {
            (buffers1._colors2, buffers2._colors2) = (buffers2._colors1, buffers1._colors1);
            (buffers1._floats2, buffers2._floats2) = (buffers2._floats1, buffers1._floats1);
        }

        /// <summary>
        /// Allocate buffers for the inputs, outputs and gradients.
        /// </summary>
        /// <param name="batchSize">The number of elements in a single batch.</param>
        public void Allocate(int batchSize)
        {
            int dimensions = _outputAllocationPairs.MaxBy(x => x.dimensions).dimensions;
            _colors1 = new ArrayView<Color>[dimensions, batchSize];
            _floats1 = new ArrayView<float>[dimensions, batchSize];

            for (int i = 0; i < dimensions; i++)
            {
                int area = _outputAllocationPairs.Where(x => x.dimensions > i).MaxBy(x => x.area).area;
                for (int j = 0; j < batchSize; j++)
                {
                    var buffer = Utility.Accelerator.Allocate1D<Color>(area);
                    _colors1[i, j] = new ArrayView<Color>(buffer, 0, area);
                    _floats1[i, j] = new ArrayView<float>(buffer, 0, 3 * area);
                }
            }
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