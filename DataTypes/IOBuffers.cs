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
        private ArrayView<float> View { get; set; }
        private int _maxLength = 0;
        private MemoryBuffer1D<float, Stride1D.Dense> _buffer;

        public ArrayView<float> InGradient => View;
        public ArrayView<float> Input => Compliment.View;
        public ArrayView<float> OutGradient => Compliment.View;
        public ArrayView<float> Output => View;
        public ArrayView<float> Gradient => Compliment.View;

        private bool _allocated = false;

        public IOBuffers Compliment { get; private set; }

        /// <summary>
        /// Sets to <see cref="IOBuffers"/> to be reflections of eachother. Aka, the input of one is the output of the other.
        /// </summary>
        /// <param name="buffers1">The first <see cref="IOBuffers"/>.</param>
        /// <param name="buffers2">The second <see cref="IOBuffers"/>.</param>
        public static void SetCompliment(IOBuffers buffers1, IOBuffers buffers2)
        {
            buffers1.Compliment = buffers2;
            buffers2.Compliment = buffers1;
        }

        /// <summary>
        /// Allocate buffers for the inputs, outputs and gradients.
        /// </summary>
        /// <param name="batchSize">The number of elements in a single batch.</param>
        public void Allocate(int batchSize)
        {
            if (_allocated)
                return;

            _buffer?.Dispose();

            _buffer = GPUManager.Accelerator.Allocate1D<float>(_maxLength * batchSize);
            View = new ArrayView<float>(_buffer, 0, _maxLength * batchSize);
            GPUManager.AddExternalMemoryUsage(4 * _maxLength * batchSize);
            _allocated = true;
        }

        /// <summary>
        /// Adds a new pair detailing the maximum number of dimensions needed for the buffers, and the maximum space needed to be allocated for those dimensions.
        /// </summary>
        /// <param name="dimensions"></param>
        /// <param name="area"></param>
        public void OutputDimensionArea(int length)
        {
            if (length > _maxLength)
            {
                _maxLength = length;
                _allocated = false;
            }
        }
    }
}