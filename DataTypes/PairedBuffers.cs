using ConvolutionalNeuralNetwork.GPU;
using ConvolutionalNeuralNetwork.Layers;
using ILGPU;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="PairedBuffers"/> class contains a set of <see cref="MemoryBuffer"/>s for use with an <see cref="ILGPU"/> kernel.
    /// Because the output of one <see cref="Layer"/> is the input of another layer, two set of these can be used such that the device inputs of one
    /// is the device outputs of another. This saves a significant amount of time instead of allocating and then de-allocating <see cref="MemoryBuffer"/>s.
    /// </summary>
    public class PairedBuffers
    {
        private bool _allocated;
        private MemoryBuffer1D<float, Stride1D.Dense> _buffer;
        private int _maxLength;

        /// <value>The complimentary <see cref="PairedBuffers"/> that shares two <see cref="ArrayView{T}"/>s.</value>
        public PairedBuffers Compliment { get; private set; }

        /// <value>The <see cref="ArrayView{T}"/> of both the incoming and outgoing gradient, for <see cref="IReflexiveLayer"/>s.</value>
        public ArrayView<float> Gradient => Compliment.View;

        /// <value>The <see cref="ArrayView{T}"/> of the incoming gradient.</value>
        public ArrayView<float> InGradient => View;

        /// <value>The <see cref="ArrayView{T}"/> of the input.</value>
        public ArrayView<float> Input => Compliment.View;

        /// <value>The <see cref="ArrayView{T}"/> of the outgoing gradient.</value>
        public ArrayView<float> OutGradient => Compliment.View;

        /// <value>The <see cref="ArrayView{T}"/> of the output.</value>
        public ArrayView<float> Output => View;
        
        private ArrayView<float> View { get; set; }

        /// <summary>
        /// Sets to <see cref="PairedBuffers"/> to be reflections of each other. Aka, the input of one is the output of the other.
        /// </summary>
        /// <param name="buffers1">The first <see cref="PairedBuffers"/>.</param>
        /// <param name="buffers2">The second <see cref="PairedBuffers"/>.</param>
        public static void SetCompliment(PairedBuffers buffers1, PairedBuffers buffers2)
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
            _allocated = true;
        }

        /// <summary>
        /// Sets the maximum space needed to be allocated for an operation.
        /// </summary>
        /// <param name="length">The required length of the <see cref="PairedBuffers"/>.</param>
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