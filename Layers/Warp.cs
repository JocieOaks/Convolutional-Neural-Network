using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    public class Warp : Layer
    {
        private Vector _inputCopy;

        [JsonConstructor] public Warp() : base(1, 1) { }

        public override string Name => "Warp Layer";

        public override void Backwards(int batchSize, bool update)
        {
            _buffers.OutGradient.SubView(0, batchSize * _inputShape.Volume).MemSetToZero();

            Index3D index = new(_inputShape.Area, _inputShape.Dimensions - 2, batchSize);
            s_backwardsAction(index, _buffers.InGradient, _inputCopy.GetArrayView<float>(),_buffers.OutGradient, _inputShape, _outputShape);

            Synchronize();

            _inputCopy.DecrementLiveCount();
        }

        public override void Forward(int batchSize)
        {
            Index1D copyIndex = new(batchSize * _inputShape.Volume);
            GPUManager.CopyAction(copyIndex, _buffers.Input, _inputCopy.GetArrayViewEmpty<float>());

            Index3D index = new(_inputShape.Area, _inputShape.Dimensions - 2, batchSize);
            s_forwardAction(index, _buffers.Input, _buffers.Output, _inputShape, _outputShape);

            Synchronize();

            _inputCopy.DecrementLiveCount();
        }

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, Shape, Shape> s_forwardAction
            = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, Shape, Shape>(WarpKernel);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, Shape, Shape> s_backwardsAction
            = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, Shape, Shape>(WarpGradientKernel);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="index">
        /// X - Batch Index
        /// Y - Dimension
        /// Z - Position Index</param>
        /// <param name=""></param>
        private static void WarpKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, Shape inputShape, Shape outputShape)
        {
            int outputOffset = outputShape.GetOffset(index.Z, index.Y);
            int inputOffset = inputShape.GetOffset(index.Z, index.Y + 2);      //The first two dimensions are the x and y warp components.
            int xOffset = inputShape.GetOffset(index.Z, 0);
            int yOffset = inputShape.GetOffset(index.Z, 1);


            float x = input[xOffset + index.X];
            float y = input[yOffset + index.X];
            int x1 = (int)XMath.Floor(x);
            int y1 = (int)XMath.Floor(y);
            int x2 = x1 + 1;
            int y2 = y1 + 1;




            float sum = 0;
            for(int i = 0; i < 2; i++)
            {
                for(int j = 0; j < 2; j++)
                {
                    float width = i == 0 ? x2 - x : x - x1;
                    float length = j == 0 ? y2 - y : y - y1;

                    if(outputShape.TryGetIndex(index.X, x1 + i, y1 + j, out int mapIndex))
                    {
                        sum += width * length * input[mapIndex + inputOffset];
                    }
                }
            }

            output[outputOffset + index.X] = sum;
        }

        private static void WarpGradientKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> input, ArrayView<float> outGradient, Shape inputShape, Shape outputShape)
        {
            int outputOffset = outputShape.GetOffset(index.Z, index.Y);
            int inputOffset = inputShape.GetOffset(index.Z, index.Y + 2);   //The first two dimensions are the x and y warp components.
            int xOffset = inputShape.GetOffset(index.Z, 0);
            int yOffset = inputShape.GetOffset(index.Z, 1);


            float x = input[xOffset + index.X];
            float y = input[yOffset + index.X];
            int x1 = (int)XMath.Floor(x);
            int y1 = (int)XMath.Floor(y);
            int x2 = x1 + 1;
            int y2 = y1 + 1;


            float dL = inGradient[outputOffset + index.X];
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    float width = i == 0 ? x2 - x : x - x1;
                    float length = j == 0 ? y2 - y : y - y1;

                    if (inputShape.TryGetIndex(index.X, x1 + i, y1 + j, out int mapIndex))
                    {
                        Atomic.Add(ref outGradient[mapIndex + inputOffset], width * length * dL);
                        Atomic.Add(ref outGradient[xOffset + index.X], (i == 0 ? -1 : 1) * length * input[mapIndex + inputOffset] * dL);
                        Atomic.Add(ref outGradient[yOffset + index.X], (j == 0 ? -1 : 1) * width * input[mapIndex + inputOffset] * dL);
                    }
                }
            }
        }

        public override Shape Startup(Shape inputShape, PairedBuffers buffers, int maxBatchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            BaseStartup(inputShape, buffers, maxBatchSize);

            _inputCopy = new Vector(maxBatchSize *  inputShape.Volume);
            _outputShape = new Shape(inputShape.Width, inputShape.Length, inputShape.Dimensions - 2);

            return _outputShape;
        }
    }
}
