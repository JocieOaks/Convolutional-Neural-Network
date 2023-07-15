using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="SkipConnectionSplit"/> class is a <see cref="Layer"/> that creates two sets of the same <see cref="FeatureMap"/>s, sending
    /// one as input to the next <see cref="Layer"/> and sending one to a <see cref="SkipConnectionConcatenate"/> later in the <see cref="Network"/>.
    /// </summary>
    public class SkipConnectionSplit : Layer, IStructuralLayer
    {
        /// <inheritdoc/>
        public override string Name => "Skip Connection Layer";

        private FeatureMap[,] _inGradientSecondary;
        private SkipConnectionConcatenate _concatenationLayer;

        /// <summary>
        /// Initializes a new instance of the <see cref="SkipConnectionSplit"/> class.
        /// </summary>
        public SkipConnectionSplit() : base(1, 1)
        {
        }

        /// <summary>
        /// Gives the corresponding <see cref="SkipConnectionConcatenate"/> layer that connects to this <see cref="SkipConnectionSplit"/>, creating
        /// it if it does not already exist.
        /// </summary>
        /// <returns>Returns the <see cref="SkipConnectionConcatenate"/>.</returns>
        public SkipConnectionConcatenate GetConcatenationLayer()
        {
            if (_concatenationLayer == null)
                _concatenationLayer = new SkipConnectionConcatenate();
            return _concatenationLayer;
        }

        private static readonly Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>> s_backwardsAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>>(BackwardsKernel);

        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index2D index = new(Infos(i).Area, 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    s_backwardsAction(index, _buffers.InGradientsColor[i, j], _inGradientSecondary[i, j].GetArrayView<Color>(), _buffers.OutGradientsFloat[i, j]);
                }
            }
            Synchronize();
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index1D index = new(Infos(i).Area);
                for (int j = 0; j < _batchSize; j++)
                {
                    GPU.GPUManager.CopyAction(index, _buffers.InputsColor[i, j], _outputs[i, j].GetArrayViewEmpty<Color>());
                    GPU.GPUManager.CopyAction(index, _buffers.InputsColor[i, j], _buffers.OutputsColor[i, j]);
                }
            }

            Synchronize();
            DecrementCacheabble(_outputs);
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override FeatureMap[,] Startup(FeatureMap[,] inputs, IOBuffers buffers)
        {
            _outputDimensions = _inputDimensions = inputs.GetLength(0);
            _buffers = buffers;

            _batchSize = (uint)inputs.GetLength(1);
            _layerInfos = new ILayerInfo[_inputDimensions];
            for (int i = 0; i < _inputDimensions; i++)
            {
                _layerInfos[i] = new StaticLayerInfo()
                {
                    Width = inputs[i, 0].Width,
                    Length = inputs[i, 0].Length,
                };
            }

            _outputs = inputs;

            _inGradientSecondary = new FeatureMap[_outputDimensions, _batchSize];

            _concatenationLayer.Connect(_outputs, _inGradientSecondary);

            return inputs;
        }

        private static void BackwardsKernel(Index2D index, ArrayView<Color> inGradient1, ArrayView<Color> inGradient2, ArrayView<float> outGradient)
        {
            outGradient[index.X * 3 + index.Y] = inGradient1[index.X][index.Y] + inGradient2[index.X][index.Y];
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