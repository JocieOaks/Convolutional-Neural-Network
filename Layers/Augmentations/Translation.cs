using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Augmentations
{

    /// <summary>
    /// The <see cref="Translation"/> layer is an augmentation that translates the input images in a random direction between 1/8 and -1/8 of the original image size, padding with zeros.
    /// </summary>
    public class Translation : Layer
    {
        private int _maxTranslationX;
        private int _maxTranslationY;

        private (int x, int y) _translation;

        /// <inheritdoc />
        public override string Name => "Translation Augment";

        [JsonConstructor]
        public Translation() : base(1, 1) { }

        /// <inheritdoc />
        public override void Backwards(int batchSize, bool update)
        {
            Index3D index = new(_inputShape.Area, _inputShape.Dimensions, batchSize);
            s_translateAction(index, _buffers.InGradient, _buffers.OutGradient, _inputShape, -_translation.x, -_translation.y);
            Synchronize();
        }

        /// <inheritdoc />
        public override void Forward(int batchSize)
        {
            _translation = (Utility.Random.Next(-_maxTranslationX, _maxTranslationX),
                Utility.Random.Next(-_maxTranslationY, _maxTranslationY));
            Index3D index = new(_inputShape.Area, _inputShape.Dimensions, batchSize);
            s_translateAction(index, _buffers.Input, _buffers.Output, _inputShape, _translation.x, _translation.y);
            Synchronize();
        }

        /// <inheritdoc />
        public override TensorShape Startup(TensorShape inputShape, PairedBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            BaseStartup(inputShape, buffers);
            _maxTranslationX = inputShape.Width / 8;
            _maxTranslationY = inputShape.Length / 8;

            return _outputShape;
        }

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, TensorShape, int, int> s_translateAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, TensorShape, int, int>(TranslateKernel);

        private static void TranslateKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, TensorShape shape, int x, int y)
        {
            int offset = shape.GetOffset(index.Z, index.Y);

            if (shape.TryGetIndex(index.X, x, y, out int mapIndex))
            {
                output[offset + index.X] = input[mapIndex + offset];
            }
            else
            {
                output[offset + index.X] = 0;
            }
        }
    }
}
