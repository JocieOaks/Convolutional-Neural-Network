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
        [JsonProperty] private SkipConnectionConcatenate _concatenationLayer;
        private MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceInGradientsSecondary;

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

        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            Accelerator accelerator = Utility.Accelerator;

            var backwardsKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>>(BackwardsKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index2D index = new(Infos(i).Area, 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);
                    _deviceInGradientsSecondary[i, j] = _inGradientSecondary[i, j].Allocate(accelerator);
                    _deviceOutGradients[i, j] = _outGradients[i, j].AllocateFloat(accelerator, false);

                    backwardsKernal(index, _deviceInGradients[i, j].View, _deviceInGradientsSecondary[i, j].View, _deviceOutGradients[i, j].View);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _outGradients[i, j].CopyFromBuffer(_deviceOutGradients[i, j]);
                    _deviceOutGradients[i, j].Dispose();
                    _deviceInGradients[i, j].Dispose();
                    _deviceInGradientsSecondary[i, j].Dispose();
                }
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            //Forward propagation does nothing, because both following layers already have a reference to the previous layers output after StartUp.
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] inputs, FeatureMap[,] outGradients)
        {
            _outputDimensions = _inputDimensions = inputs.GetLength(0);

            _batchSize = inputs.GetLength(1);
            _layerInfos = new ILayerInfo[_inputDimensions];
            _inputs = _outputs = inputs;

            _outGradients = outGradients;

            _inGradients = new FeatureMap[_outputDimensions, _batchSize];
            _inGradientSecondary = new FeatureMap[_outputDimensions, _batchSize];

            for (int i = 0; i < _inputDimensions; i++)
            {
                ILayerInfo layer;
                layer = _layerInfos[i] = new StaticLayerInfo()
                {
                    Width = inputs[i, 0].Width,
                    Length = inputs[i, 0].Length,
                };

                for (int j = 0; j < _batchSize; j++)
                {
                    _outGradients[i, j] = new FeatureMap(layer.InputWidth, layer.InputLength);
                }
            }

            _deviceInGradients = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions, _batchSize];
            _deviceOutGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions, _batchSize];
            _deviceInGradientsSecondary = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions, _batchSize];

            _concatenationLayer.Connect(inputs, _inGradientSecondary);

            return (inputs, _inGradients);
        }

        private static void BackwardsKernal(Index2D index, ArrayView<Color> inGradient1, ArrayView<Color> inGradient2, ArrayView<float> outGradient)
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