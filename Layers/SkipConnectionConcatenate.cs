using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="SkipConnectionConcatenate"/> class is a <see cref="Layer"/> for combining a set of <see cref="FeatureMap"/>s from the previous
    /// <see cref="Layer"/> with the <see cref="FeatureMap"/>s from its corresponding <see cref="SkipConnectionSplit"/>.
    /// </summary>
    public class SkipConnectionConcatenate : Layer, IStructuralLayer
    {
        private FeatureMap[,] _inputsSecondary;
        private FeatureMap[,] _outGradientsSecondary;
        private MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceSecondary;

        private int _secondaryDimensions;

        /// <summary>
        /// Initializes a new instance of the <see cref="SkipConnectionConcatenate"/> class.
        /// </summary>
        public SkipConnectionConcatenate() : base(1, 1)
        {
        }

        /// <inheritdoc/>
        public override string Name => "Concatenation Layer";

        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index1D index = new(_layerInfos[i].InputArea);
                for (int j = 0; j < _batchSize; j++)
                {
                    Utility.CopyAction(index, _buffers.InGradientsColor[i, j], _buffers.OutGradientsColor[i, j]);
                }
            }

            for (int i = 0; i < _secondaryDimensions; i++)
            {
                Index1D index = new(_layerInfos[i + _inputDimensions].InputArea);
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceSecondary[i, j] = _outGradientsSecondary[i, j].AllocateEmpty();
                    Utility.CopyAction(index, _buffers.InGradientsColor[_inputDimensions + i, j], _deviceSecondary[i, j].View);
                }
            }

            Utility.Accelerator.Synchronize();

            for(int i = 0; i < _secondaryDimensions; i++)
            {
                for(int j = 0; j < _batchSize; j++)
                {
                    _outGradientsSecondary[i, j].CopyFromBuffer(_deviceSecondary[i, j]);
                    _deviceSecondary[i, j].Dispose();
                }
            }
        }

        /// <summary>
        /// Connects the <see cref="SkipConnectionConcatenate"/> with its <see cref="SkipConnectionSplit"/> sharing the <see cref="FeatureMap"/>s
        /// between them.
        /// </summary>
        /// <param name="inputs">The split outputs of the <see cref="SkipConnectionSplit"/>.</param>
        /// <param name="outGradients">The split inGradients of the <see cref="SkipConnectionSplit"/>.</param>
        public void Connect(FeatureMap[,] inputs, FeatureMap[,] outGradients)
        {
            _inputsSecondary = inputs;
            _secondaryDimensions = inputs.GetLength(0);
            _batchSize = inputs.GetLength(1);
            _outGradientsSecondary = outGradients;
            _deviceSecondary = new MemoryBuffer1D<Color, Stride1D.Dense>[_secondaryDimensions, _batchSize];

            for (int i = 0; i < _secondaryDimensions; i++)
            {
                int width = inputs[i, 0].Width;
                int length = inputs[i, 0].Length;
                for (int j = 0; j < _batchSize; j++)
                {
                    _outGradientsSecondary[i, j] = new FeatureMap(width, length);
                }
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index1D index = new(_layerInfos[i].InputArea);
                for (int j = 0; j < _batchSize; j++)
                {
                    Utility.CopyAction(index, _buffers.InputsColor[i, j], _buffers.OutputsColor[i, j]);
                }
            }

            for (int i = 0; i < _secondaryDimensions; i++)
            {
                Index1D index = new(_layerInfos[i + _inputDimensions].InputArea);
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceSecondary[i, j] = _inputsSecondary[i, j].Allocate();
                    Utility.CopyAction(index, _deviceSecondary[i, j].View, _buffers.OutputsColor[_inputDimensions + i, j]);
                }
            }

            Utility.Accelerator.Synchronize();

            for (int i = 0; i < _secondaryDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceSecondary[i, j].Dispose();
                }
            }
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override FeatureMap[,] Startup(FeatureMap[,] inputs, IOBuffers buffers)
        {
            _inputDimensions = inputs.GetLength(0);
            _outputDimensions = _inputDimensions + _secondaryDimensions;

            _outputs = new FeatureMap[_outputDimensions, _batchSize];

            _layerInfos = new ILayerInfo[_inputDimensions + _secondaryDimensions];
            for (int i = 0; i < _inputDimensions; i++)
            {
                StaticLayerInfo layer = new StaticLayerInfo()
                {
                    Width = inputs[i, 0].Width,
                    Length = inputs[i, 0].Length,
                };

                _layerInfos[i] = layer;

                for (int j = 0; j < _batchSize; j++)
                {
                    _outputs[i, j] = new FeatureMap(layer.Width, layer.Length);
                }
            }

            for (int i = 0; i < _secondaryDimensions; i++)
            {
                StaticLayerInfo layer = new StaticLayerInfo()
                {
                    Width = _inputsSecondary[i, 0].Width,
                    Length = _inputsSecondary[i, 0].Length
                };
                _layerInfos[i + _inputDimensions] = layer;
                for (int j = 0; j < _batchSize; j++)
                {
                    _outputs[_inputDimensions + i, j] = new FeatureMap(layer.Width, layer.Length);
                }
            }

            _buffers = buffers;
            for (int i = 0; i < _outputDimensions; i++)
                buffers.OutputDimensionArea(i, _outputs[i, 0].Area);

            return _outputs;
        }
    }
}