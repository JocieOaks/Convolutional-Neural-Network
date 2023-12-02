using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers.Augmentations
{
    public class Cutout : Layer, IUnchangedLayer
    {
        private int _halfWidth;
        private int _halfLength;
        private int _fourthWidth;
        private int _fourthLength;

        private Index3D _index;
        private int _offsetX;
        private int _offsetY;

        [JsonConstructor]
        public Cutout() : base(1, 1) { }

        /// <inheritdoc />
        public override string Name => "Cutout Augmentation";

        /// <inheritdoc />
        public override void Backwards(int batchSize, bool update)
        {
            s_cutoutAction(_index, _buffers.Gradient, _inputShape, _offsetX, _offsetY);
            Synchronize();
        }

        /// <inheritdoc />
        public override void Forward(int batchSize)
        {
            int baseOffsetX = Utility.Random.Next(0, _inputShape.Width);
            int baseOffsetY = Utility.Random.Next(0, _inputShape.Length);

            _offsetX = baseOffsetX - _fourthWidth;
            int width = _halfWidth;
            if (_offsetX < 0)
            {
                width += _offsetX;
                _offsetX = 0;
            }
            else if (_offsetX + width > _inputShape.Width)
            {
                width = _inputShape.Width - _offsetX;
            }

            _offsetY = baseOffsetY - _fourthLength;
            int length = _halfLength;
            if (_offsetY < 0)
            {
                length += _offsetY;
                _offsetY = 0;
            }
            else if (_offsetY + length > _inputShape.Length)
            {
                length = _inputShape.Length - _offsetY;
            }

            _index = new Index3D(width, length, batchSize * _inputShape.Dimensions);


            s_cutoutAction(_index, _buffers.Input, _inputShape, _offsetX, _offsetY);
            Synchronize();
        }

        /// <inheritdoc />
        public override Shape Startup(Shape inputShape, IOBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            BaseStartup(inputShape, buffers);
            _halfWidth = inputShape.Width / 2;
            _fourthWidth = inputShape.Width / 4;

            _halfLength = inputShape.Length / 2;
            _fourthLength = inputShape.Length / 4;

            return _outputShape;
        }

        private static readonly Action<Index3D, ArrayView<float>, Shape, int, int> s_cutoutAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, Shape, int, int>(CutoutKernel);

        private static void CutoutKernel(Index3D index, ArrayView<float> input, Shape shape, int x, int y)
        {
            int offset = index.Z * shape.Area;

            int inputIndex = index.X + x + (index.Y + y) * shape.Width;

            input[offset + inputIndex] = 0;
        }
    }
}
