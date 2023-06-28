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
        private MemoryBuffer1D<StaticLayerInfo, Stride1D.Dense>[] _deviceInfos;

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
                for (int j = 0; j < _batchSize; j++)
                {
                    for (int y = 0; y < _inputs[i, j].Length; y++)
                    {
                        for (int x = 0; x < _inputs[i, j].Width; x++)
                        {
                            _outGradients[i, j][x, y] = _inputs[i, j][x, y].ReLUPropagation() * _inGradients[i, j][x, y];
                        }
                    }
                }
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    for(int y = 0; y < _inputs[i,j].Length; y++)
                    {
                        for(int x = 0; x < _inputs[i,j].Width; x++)
                        {
                            _outputs[i, j][x, y] = _inputs[i, j][x, y].ReLU();
                        }
                    }
                }
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
            _deviceInfos = new MemoryBuffer1D<StaticLayerInfo, Stride1D.Dense>[_inputDimensions];
            return (_outputs, _inGradients);
        }

        private static void BackwardsKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> inGradient, ArrayView<float> outGradient, ArrayView<StaticLayerInfo> info)
        {
            int mapsIndex = info[0].Index(index.X, index.Y);
            outGradient[3 * mapsIndex + index.Z] = input[mapsIndex].ReLUPropagation()[index.Z] * inGradient[mapsIndex][index.Z];
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