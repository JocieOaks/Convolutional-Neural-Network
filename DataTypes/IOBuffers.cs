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
        private ArrayView<float> _view1;
        private ArrayView<float> _view2;
        private int _maxLength = 0;
        private MemoryBuffer1D<float, Stride1D.Dense> _buffer;

        public ArrayView<float> InGradient => _view1;
        public ArrayView<float> Input => _view2;
        public ArrayView<float> OutGradient => _view2;
        public ArrayView<float> Output => _view1;
        public ArrayView<float> Gradient => _view2;

        /// <summary>
        /// Sets to <see cref="IOBuffers"/> to be reflections of eachother. Aka, the input of one is the output of the other.
        /// </summary>
        /// <param name="buffers1">The first <see cref="IOBuffers"/>.</param>
        /// <param name="buffers2">The second <see cref="IOBuffers"/>.</param>
        public static void SetCompliment(IOBuffers buffers1, IOBuffers buffers2)
        {
            (buffers1._view2, buffers2._view2) = (buffers2._view1, buffers1._view1);
        }

        /// <summary>
        /// Allocate buffers for the inputs, outputs and gradients.
        /// </summary>
        /// <param name="batchSize">The number of elements in a single batch.</param>
        public void Allocate(int batchSize)
        {
            _buffer?.Dispose();

            _buffer = GPUManager.Accelerator.Allocate1D<float>(_maxLength * batchSize);
            _view1 = new ArrayView<float>(_buffer, 0, _maxLength * batchSize);
            GPUManager.AddExternalMemoryUsage(4 * _maxLength * batchSize);
        }

        /// <summary>
        /// Adds a new pair detailing the maximum number of dimensions needed for the buffers, and the maximum space needed to be allocated for those dimensions.
        /// </summary>
        /// <param name="dimensions"></param>
        /// <param name="area"></param>
        public void OutputDimensionArea(int length)
        {
            if(length > _maxLength)
                _maxLength = length;
        }
    }
}