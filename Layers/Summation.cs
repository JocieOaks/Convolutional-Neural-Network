using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers
{

    /// <summary>
    /// The <see cref="Summation"/> class is a <see cref="Layer"/> that sums the <see cref="FeatureMap"/>s across multiple dimensions, reducing the number of
    /// dimensions in the <see cref="Network"/>.
    /// Note: When summing <see cref="FeatureMap"/>s that have been batch normalized to have a mean of 0, the mean will remain the same, but the standard deviation
    /// will change.
    /// </summary>
    [Serializable]
    public class Summation : Layer, IStructuralLayer
    {
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<StaticLayerInfo>> s_backwardsAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<StaticLayerInfo>>(BackwardsKernel);

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<StaticLayerInfo>> s_forwardAction =
            GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<StaticLayerInfo>>(ForwardKernel);

        private ArrayView<StaticLayerInfo> _deviceInfo;
        [JsonProperty] private int _dimensionDivisor;

        [JsonConstructor] public Summation() : base(1, 1) { }
        /// <inheritdoc/>
        public override string Name => "Summation Layer";
        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            Index3D index = new(_batchSize, _inputDimensions, Infos(0).Area);
            s_backwardsAction(index, _buffers.Input, _buffers.Output, _deviceInfo);

            Synchronize();
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            _buffers.Output.SubView(0, _batchSize * _outputDimensions * Infos(0).Area).MemSetToZero();

            Index3D index = new(_batchSize, _inputDimensions, Infos(0).Area);
            s_forwardAction(index, _buffers.Input, _buffers.Output, _deviceInfo);

            Synchronize();
        }
        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <summary>
        /// Sets the number of dimensions for the output. The input dimensions will need to be a multiple of the output dimensions
        /// or vice versa. Overwrites <see cref="SetOutputDivisor(int)"/>.
        /// </summary>
        /// <param name="dimensions">The number of output dimensions.</param>
        public void SetOutputDimensions(int dimensions)
        {
            _outputDimensions = dimensions;
        }

        /// <summary>
        /// Sets the number of dimensions for the output as a multiple of the input dimensions.
        /// Is Overwritten by <see cref="SetOutputDimensions(int)"/>.
        /// </summary>
        /// <param name="divisor">The factor to multiply the input dimensions to set the output dimensions.
        /// Must be greater than 1.</param>
        public void SetOutputDivisor(int divisor)
        {
            if (divisor <= 1)
            {
                throw new ArgumentException("Dimension divisor must be greater than 1.");
            }
            _dimensionDivisor = divisor;
        }

        /// <inheritdoc/>
        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, int batchSize)
        {
            if (_outputDimensions != 0)
            {
                _dimensionDivisor = inputShapes.Length / _outputDimensions;
            }

            BaseStartup(inputShapes, buffers, batchSize, -_dimensionDivisor);

            _deviceInfo = GPUManager.Accelerator.Allocate1D(Array.ConvertAll(_layerInfos, info => (StaticLayerInfo)info)).View;

            return _outputShapes;
        }

        private static void BackwardsKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<StaticLayerInfo> infoView)
        {
            StaticLayerInfo info = infoView[index.Y];
            int outGradientIndex = (index.X * info.InputDimensions + index.Y) * info.Area + index.Z;
            int inGradientIndex = (index.X * info.OutputDimensions + index.Y % info.OutputDimensions) * info.Area + index.Z;

            outGradient[outGradientIndex] = inGradient[inGradientIndex];
        }

        private static void ForwardKernel(Index3D index, ArrayView<float> input, ArrayView<float> output, ArrayView<StaticLayerInfo> infoView)
        {
            StaticLayerInfo info = infoView[index.Y];
            int inputIndex = (index.X * info.InputDimensions + index.Y) * info.Area + index.Z;
            int outputIndex = (index.X * info.OutputDimensions + index.Y % info.OutputDimensions) * info.Area + index.Z;

            Atomic.Add(ref output[outputIndex], input[inputIndex]);
        }
        /// <summary>
        /// Gets the <see cref="StaticLayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="StaticLayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="StaticLayerInfo"/> corresponding to an input dimension.</returns>
        private StaticLayerInfo Infos(int index)
        {
            return (StaticLayerInfo)_layerInfos[index];
        }
    }
}
