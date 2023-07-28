using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    public class Upsampling : Layer
    {
        private Vector _inputCopy;

        [JsonConstructor] public Upsampling(int ratio) : base(1, ratio) { }

        public override string Name => "Upsampling Layer";

        public override void Backwards(int batchSize)
        {
            Index3D index = new(batchSize, _inputShape.Dimensions, _inputShape.Area);
            s_backwardsAction(index, _buffers.InGradient, _inputCopy.GetArrayView<float>(), _buffers.OutGradient, Info, batchSize);

            Synchronize();

            _inputCopy.DecrementLiveCount();
        }

        public override void Forward(int batchSize)
        {
            Index1D copyIndex = new(batchSize * _inputShape.Volume + 2 * _inputShape.Area);
            GPUManager.CopyAction(copyIndex, _buffers.Input, _inputCopy.GetArrayViewEmpty<float>());

            Index3D index = new(batchSize, _inputShape.Dimensions, _inputShape.Area);
            s_forwardAction(index, _buffers.Input, _buffers.Output, Info, batchSize);

            Synchronize();

            _inputCopy.DecrementLiveCount();
        }

        public override void Reset()
        {
        }

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, InverseLayerInfo, int> s_forwardAction
            = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, InverseLayerInfo, int>(ForwardKernel);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, InverseLayerInfo, int> s_backwardsAction
            = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, InverseLayerInfo, int>(BackwardsKernel);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="index">
        /// X - Batch Index
        /// Y - Dimension
        /// Z - Position Index</param>
        /// <param name=""></param>
        private static void ForwardKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, InverseLayerInfo info, int batchSize)
        {
            (int inputOffset, int outputOffset) = info.GetOffset(index.X, index.Y);

            /*float x = input[batchSize * shape.Volume + index.Z];
            float y = input[batchSize * shape.Volume + shape.Area + index.Z];
            int x1 = (int)x;
            int y1 = (int)y;
            int x2 = x1 + 1;
            int y2 = y1 + 1;




            float sum = 0;
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    float width = i == 0 ? x2 - x : x - x1;
                    float length = j == 0 ? y2 - y : y - y1;

                    if (shape.TryGetIndex(index.Z, x1 + i, y1 + i, out int mapIndex))
                    {
                        sum += width * length * input[mapIndex + inputOffset];
                    }
                }
            }

            output[offset + index.Z] = sum;*/
        }

        private static void BackwardsKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> input, ArrayView<float> outGradient, InverseLayerInfo info, int batchSize)
        {
            /*(int inputOffset, int outputOffset) = info.GetOffset(index.X, index.Y);

            float x = input[batchSize * shape.Volume + index.Z];
            float y = input[batchSize * shape.Volume + shape.Area + index.Z];
            int x1 = (int)x;
            int y1 = (int)y;
            int x2 = x1 + 1;
            int y2 = y1 + 1;


            float dL = inGradient[offset + index.Z];
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    float width = i == 0 ? x2 - x : x - x1;
                    float length = j == 0 ? y2 - y : y - y1;

                    if (shape.TryGetIndex(index.Z, x1 + i, y1 + i, out int mapIndex))
                    {
                        Atomic.Add(ref outGradient[mapIndex + inputOffset], width * length * dL);
                        Atomic.Add(ref outGradient[batchSize * shape.Volume + index.Z], (i == 0 ? -1 : 1) * length * input[mapIndex + inputOffset] * dL);
                        Atomic.Add(ref outGradient[batchSize * shape.Volume + shape.Area + index.Z], (j == 0 ? -1 : 1) * width * input[mapIndex + inputOffset] * dL);
                    }
                }
            }*/
        }

        public override Shape Startup(Shape inputShape, IOBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            BaseStartup(inputShape, buffers, maxBatchSize);


            _outputShape = new Shape(2 * inputShape.Width, 2 * inputShape.Length, inputShape.Dimensions);
            _layerInfo = new InverseLayerInfo(inputShape, _outputShape, _filterSize, _stride);

            _inputCopy = new Vector(maxBatchSize * inputShape.Volume);
            

            return _outputShape;
        }

        private InverseLayerInfo Info => (InverseLayerInfo)_layerInfo;
    }
}
