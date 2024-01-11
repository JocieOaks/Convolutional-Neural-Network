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
            Buffers.OutGradient.SubView(0, batchSize * InputShape.Volume).MemSetToZero();
            Index3D index = new(OutputShape.Area, InputShape.Dimensions, batchSize);
            s_backwardsAction(index, Buffers.InGradient, Buffers.OutGradient, Info);

            Synchronize();
        }

        public override void Forward(int batchSize)
        {
            Index3D index = new(OutputShape.Area, InputShape.Dimensions, batchSize);
            s_forwardAction(index, Buffers.Input, Buffers.Output, Info);

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
            if (Ready)
                return OutputShape;
            Ready = true;

            BaseStartup(inputShape, buffers, maxBatchSize);


            OutputShape = new TensorShape(Stride * inputShape.Width, Stride * inputShape.Length, inputShape.Dimensions);
            LayerInfo = new LayerInfo(inputShape, OutputShape, FilterSize, Stride);
            buffers.OutputDimensionArea(OutputShape.Volume);

            return OutputShape;
        }

        private LayerInfo Info => (LayerInfo)LayerInfo;
    }
}
