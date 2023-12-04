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

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo> s_forwardAction
            = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo>(ForwardUpKernel);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo> s_backwardsAction
            = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo>(BackwardsKernel);

        private static void ForwardUpKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, LayerInfo info)
        {
            (int inputOffset, int outputOffset) = info.GetOffset(index.Z, index.Y);

            (int x1, int y1) = GetInputCoordinates(info, index.X, out float x, out float y);

            int x2 = x1 + 1;
            int y2 = y1 + 1;

            info.TryGetContractionIndex(index.X, 0, 0, out int baseIndex);
            float x0y0 = input[baseIndex + inputOffset];



            float sum = 0;
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    float width = i == 0 ? x2 - x : x - x1;
                    float length = j == 0 ? y2 - y : y - y1;

                    if (info.TryGetContractionIndex(index.X, i, j, out int inputIndex))
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

        private static void BackwardsKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, LayerInfo info)
        {
            (int outGradientOffset, int inGradientOffset) = info.GetOffset(index.Z, index.Y);

            (int x1, int y1) = GetInputCoordinates(info, index.X, out float x, out float y);

            int x2 = x1 + 1;
            int y2 = y1 + 1;

            info.TryGetContractionIndex(index.X, 0, 0, out int baseIndex);

            float dL = inGradient[index.X + inGradientOffset];

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    float width = i == 0 ? x2 - x : x - x1;
                    float length = j == 0 ? y2 - y : y - y1;

                    if (info.TryGetContractionIndex(index.X, i, j, out int inputIndex))
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

        private static (int, int) GetInputCoordinates(LayerInfo info, int outputIndex, out float xFloat, out float yFloat)
        {
            int x = outputIndex % info.ExpansionWidth;
            int y = outputIndex / info.ExpansionWidth;

            x += info.Padding;
            y += info.Padding;

            xFloat = (float)x / info.Stride;
            yFloat = (float)y / info.Stride;

            x /= info.Stride;
            y /= info.Stride;

            return (x, y);
        }

        public override TensorShape Startup(TensorShape inputShape, PairedBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            BaseStartup(inputShape, buffers, maxBatchSize);


            _outputShape = new TensorShape(_stride * inputShape.Width, _stride * inputShape.Length, inputShape.Dimensions);
            _layerInfo = new LayerInfo(inputShape, _outputShape, _filterSize, _stride);
            buffers.OutputDimensionArea(_outputShape.Volume);

            return _outputShape;
        }

        private LayerInfo Info => (LayerInfo)_layerInfo;
    }
}
