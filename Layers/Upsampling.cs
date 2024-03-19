using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="Upsampling"/> class is a <see cref="Layer"/> that increases the scale of the input <see cref="Tensor"/>.
    /// </summary>
    public class Upsampling : Layer
    {
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo> s_backwardsAction
                    = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo>(BackwardsKernel);

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo> s_forwardAction
                    = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, LayerInfo>(ForwardUpKernel);

        /// <summary>
        /// Initializes a new instance of the <see cref="Upsampling"/> class.
        /// </summary>
        /// <param name="ratio">The amount to scale the input by.</param>
        public Upsampling(int ratio) : base(1, ratio) { }

        /// <inheritdoc />
        public override string Name => "Upsampling Layer";

        /// <inheritdoc />
        public override void Backwards(int batchSize, bool update)
        {
            Views.OutGradient.SubView(0, batchSize * InputShape.Volume).MemSetToZero();
            Index3D index = new(OutputShape.Area, InputShape.Dimensions, batchSize);
            s_backwardsAction(index, Views.InGradient, Views.OutGradient, LayerInfo);

            GPUManager.Accelerator.Synchronize();
        }

        /// <inheritdoc />
        public override void Forward(int batchSize)
        {
            Index3D index = new(OutputShape.Area, InputShape.Dimensions, batchSize);
            s_forwardAction(index, Views.Input, Views.Output, LayerInfo);

            GPUManager.Accelerator.Synchronize();
        }

        /// <inheritdoc />
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int maxBatchSize)
        {
            if (Initialized)
                return OutputShape;
            Initialized = true;

            BaseStartup(inputShape, views, maxBatchSize);


            OutputShape = new TensorShape(Stride * inputShape.Width, Stride * inputShape.Length, inputShape.Dimensions);
            LayerInfo = new LayerInfo(inputShape, OutputShape, FilterSize, Stride);
            views.OutputDimensionArea(OutputShape.Volume);

            return OutputShape;
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

        private static void ForwardUpKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, LayerInfo info)
        {
            (int inputOffset, int outputOffset) = info.GetOffset(index.Z, index.Y);

            (int x1, int y1) = GetInputCoordinates(info, index.X, out float x, out float y);

            int x2 = x1 + 1;
            int y2 = y1 + 1;

            info.TryGetContractionIndex(index.X, 0, 0, out int baseIndex);
            float origin = input[baseIndex + inputOffset];



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
                        sum += width * length * origin;
                    }
                }
            }

            output[index.X + outputOffset] = sum;
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
    }
}
