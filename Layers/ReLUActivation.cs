using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="ReLUActivation"/> class is a <see cref="Layer"/> is an activation to add non-linearity to the <see cref="Network"/>.
    /// </summary>
    [Serializable]
    public class ReLUActivation : Layer, ISecondaryLayer
    {
        private static readonly Action<Index3D, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<StaticLayerInfo>> s_backwardsAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<StaticLayerInfo>>(BackwardsKernal);
        private static readonly Action<Index2D, ArrayView<Color>, ArrayView<int>, ArrayView<Color>, ArrayView<StaticLayerInfo>> s_forwardAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<int>, ArrayView<Color>, ArrayView<StaticLayerInfo>>(ForwardKernal);
        private MemoryBuffer1D<StaticLayerInfo, Stride1D.Dense>[] _deviceInfos;
        private MemoryBuffer1D<int, Stride1D.Dense>[,] _deviceZeroed;
        private int[,][] _zeroed;

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
            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = Utility.Accelerator.Allocate1D(new StaticLayerInfo[] { Infos(i) });
                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceZeroed[i, j] = Utility.Accelerator.Allocate1D(_zeroed[i, j]);
                    s_backwardsAction(index, _deviceZeroed[i, j].View, _buffers.InGradientsFloat[i, j], _buffers.OutGradientsFloat[i, j], _deviceInfos[i].View);
                }
            }

            Utility.Accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for(int j = 0; j < _batchSize; j++)
                {
                    _deviceZeroed[i, j].Dispose();
                }
                _deviceInfos[i].Dispose();
            }
        }
        /// <inheritdoc/>
        public override void Forward()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = Utility.Accelerator.Allocate1D(new StaticLayerInfo[] { Infos(i) });
                Index2D index = new(Infos(i).Width, Infos(i).Length);

                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceZeroed[i, j] = Utility.Accelerator.Allocate1D(_zeroed[i, j]);

                    s_forwardAction(index, _buffers.InputsColor[i, j], _deviceZeroed[i, j].View, _buffers.OutputsColor[i, j], _deviceInfos[i].View);
                }
            }

            Utility.Accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for(int j = 0; j < _batchSize; j++)
                {
                    _deviceZeroed[i, j].Dispose();
                }
                _deviceInfos[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override FeatureMap[,] Startup(FeatureMap[,] inputs, IOBuffers buffers)
        {
            BaseStartup(inputs, buffers);
            _zeroed = new int[_inputDimensions, _batchSize][];
            for(int i = 0; i < _inputDimensions; i++)
            {
                for(int j = 0; j < _batchSize; j++)
                {
                    _zeroed[i, j] = new int[inputs[i, j].Area * 3];
                }
            }

            _deviceInfos = new MemoryBuffer1D<StaticLayerInfo, Stride1D.Dense>[_inputDimensions];
            _deviceZeroed = new MemoryBuffer1D<int, Stride1D.Dense>[_inputDimensions, _batchSize];
            return _outputs;
        }

        private static void BackwardsKernal(Index3D index, ArrayView<int> zeroed, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<StaticLayerInfo> info)
        {
            int mapsIndex = info[0].Index(index.X, index.Y);
            outGradient[3 * mapsIndex + index.Z] = zeroed[3 * mapsIndex + index.Z] == 1 ? 0 : inGradient[3 * mapsIndex + index.Z];
        }

        private static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<int> zeroed, ArrayView<Color> output, ArrayView<StaticLayerInfo> info)
        {
            int mapsIndex = info[0].Index(index.X, index.Y);
            for(int i = 0; i < 3; i++)
            {
                zeroed[3 * mapsIndex + i] = input[mapsIndex][i] < 0 ? 1 : 0;
            }
            output[mapsIndex] = input[mapsIndex].ReLU();
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