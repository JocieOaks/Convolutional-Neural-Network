using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    public class Upsampling : Layer
    {
        [JsonConstructor] public Upsampling(int ratio) : base(1, ratio) { }

        public override string Name => "Upsampling Layer";

        public override void Backwards(int batchSize, bool update)
        {
            _buffers.OutGradient.SubView(0, batchSize * _inputShape.Volume).MemSetToZero();
            Index3D index = new(_outputShape.Area, _inputShape.Dimensions, batchSize);
            s_backwardsAction(index, _buffers.InGradient, _buffers.OutGradient, Info);

            Synchronize();
        }

        public override void Forward(int batchSize)
        {
            Index3D index = new(_outputShape.Area, _inputShape.Dimensions, batchSize);
            s_forwardAction(index, _buffers.Input, _buffers.Output, Info);

            Synchronize();
        }

        public override void Reset()
        {
        }

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, InverseLayerInfo> s_forwardAction
            = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, InverseLayerInfo>(ForwardUpKernel);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, InverseLayerInfo> s_backwardsAction
            = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, InverseLayerInfo>(BackwardsKernel);

        private static void ForwardUpKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, InverseLayerInfo info)
        {
            (int inputOffset, int outputOffset) = info.GetOffset(index.Z, index.Y);

            (int x1, int y1) = info.GetInputCoordinates(index.X, out float x, out float y);

            int x2 = x1 + 1;
            int y2 = y1 + 1;

            info.TryGetInputIndex(x1, y1, 0, 0, out int baseIndex);
            float x0y0 = input[baseIndex + inputOffset];



            float sum = 0;
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    float width = i == 0 ? x2 - x : x - x1;
                    float length = j == 0 ? y2 - y : y - y1;

                    if (info.TryGetInputIndex(x1, y1, i, j, out int inputIndex))
                    {
                        sum += width * length * input[inputIndex + inputOffset];
                    }
                    else
                    {
                        sum += width * length * x0y0;
                    }
                }
            }

            output[index.X + outputOffset] = sum;
        }

        private static void BackwardsKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, InverseLayerInfo info)
        {
            (int outGradientOffset, int inGradientOffset) = info.GetOffset(index.Z, index.Y);

            (int x1, int y1) = info.GetInputCoordinates(index.X, out float x, out float y);

            int x2 = x1 + 1;
            int y2 = y1 + 1;

            info.TryGetInputIndex(x1, y1, 0, 0, out int baseIndex);

            float dL = inGradient[index.X + inGradientOffset];

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    float width = i == 0 ? x2 - x : x - x1;
                    float length = j == 0 ? y2 - y : y - y1;

                    if (info.TryGetInputIndex(x1, y1, i, j, out int inputIndex))
                    {
                        Atomic.Add(ref outGradient[inputIndex + outGradientOffset], width * length * dL);
                    }
                    else
                    {
                        Atomic.Add(ref outGradient[baseIndex + outGradientOffset], width * length * dL);
                    }
                }
            }
        }

        public override Shape Startup(Shape inputShape, IOBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            BaseStartup(inputShape, buffers, maxBatchSize);


            _outputShape = new Shape(_stride * inputShape.Width, _stride * inputShape.Length, inputShape.Dimensions);
            _layerInfo = new InverseLayerInfo(inputShape, _outputShape, _filterSize, _stride);
            buffers.OutputDimensionArea(_outputShape.Volume);

            return _outputShape;
        }

        private InverseLayerInfo Info => (InverseLayerInfo)_layerInfo;
    }
}
