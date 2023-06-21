using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="ReLUActivation"/> class is a <see cref="Layer"/> is an activation to add non-linearity to the <see cref="Network"/>.
    /// </summary>
    [Serializable]
    public class ReLUActivation : Layer, ISecondaryLayer
    {
        private MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense>[] _deviceInfos;

        /// <summary>
        /// Initializes a new instance of the <see cref="ReLUActivation"/> class.
        /// </summary>
        [JsonConstructor]
        public ReLUActivation() : base(1, 1)
        {
        }

        /// <inheritdoc/>
        public override string Name => "Activation Layer";

        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            Accelerator accelerator = ConvolutionalNeuralNetwork.Utility.Accelerator;

            var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(BackwardsKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });
                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceOutGradients[i, j] = _inputs[i, j].AllocateFloat(accelerator);
                    _deviceInputs[i, j] = _inputs[i, j].Allocate(accelerator);
                    _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);

                    forwardKernal(index, _deviceInputs[i, j].View, _deviceInGradients[i, j].View, _deviceOutGradients[i, j].View, _deviceInfos[i].View);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _outGradients[i, j].CopyFromBuffer(_deviceOutGradients[i, j]);

                    _deviceOutGradients[i, j].Dispose();
                    _deviceInputs[i, j].Dispose();
                    _deviceInGradients[i, j].Dispose();
                }

                _deviceInfos[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            Accelerator accelerator = ConvolutionalNeuralNetwork.Utility.Accelerator;

            var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<SingleLayerInfo>>(ForwardKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });
                Index2D index = new(Infos(i).Width, Infos(i).Length);

                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceOutputs[i, j] = _inputs[i, j].AllocateEmpty(accelerator);
                    _deviceInputs[i, j] = _inputs[i, j].Allocate(accelerator);

                    forwardKernal(index, _deviceInputs[i, j].View, _deviceOutputs[i, j].View, _deviceInfos[i].View);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _outputs[i, j].CopyFromBuffer(_deviceOutputs[i, j]);

                    _deviceOutputs[i, j].Dispose();
                    _deviceInputs[i, j].Dispose();
                }

                _deviceInfos[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] inputs, FeatureMap[,] outGradients)
        {
            BaseStartup(inputs, outGradients);
            _deviceInfos = new MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense>[_inputDimensions];
            return (_outputs, _inGradients);
        }

        private static void BackwardsKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> inGradient, ArrayView<float> outGradient, ArrayView<SingleLayerInfo> info)
        {
            int mapsIndex = info[0].Index(index.X, index.Y);
            outGradient[3 * mapsIndex + index.Z] = input[mapsIndex].ReLUPropogation()[index.Z] * inGradient[mapsIndex][index.Z];
        }

        private static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> output, ArrayView<SingleLayerInfo> info)
        {
            int mapsIndex = info[0].Index(index.X, index.Y);
            output[mapsIndex] = input[mapsIndex].ReLU();
        }

        /// <summary>
        /// Gets the <see cref="SingleLayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="SingleLayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="SingleLayerInfo"/> corresponding to an input dimension.</returns>
        private SingleLayerInfo Infos(int index)
        {
            return (SingleLayerInfo)_layerInfos[index];
        }
    }
}