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
        /*private FeatureMap[,] _inputsSecondary;
        private FeatureMap[,] _outGradientsSecondary;

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
                    GPU.GPUManager.CopyAction(index, _buffers.InGradient[i, j], _buffers.OutGradient[i, j]);
                }
            }

            for (int i = 0; i < _secondaryDimensions; i++)
            {
                Index1D index = new(_layerInfos[i + _inputDimensions].InputArea);
                for (int j = 0; j < _batchSize; j++)
                {
                    GPU.GPUManager.CopyAction(index, _buffers.InGradient[_inputDimensions + i, j], _outGradientsSecondary[i, j].GetArrayViewEmpty<float>());
                }
            }
            Synchronize();
            DecrementCacheabble(_outGradientsSecondary);
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
            _batchSize = (uint)inputs.GetLength(1);
            _outGradientsSecondary = outGradients;

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
                    GPU.GPUManager.CopyAction(index, _buffers.Input[i, j], _buffers.Output[i, j]);
                }
            }

            for (int i = 0; i < _secondaryDimensions; i++)
            {
                Index1D index = new(_layerInfos[i + _inputDimensions].InputArea);
                for (int j = 0; j < _batchSize; j++)
                {
                    GPU.GPUManager.CopyAction(index, _inputsSecondary[i, j].GetArrayView<float>(), _buffers.Output[_inputDimensions + i, j]);
                }
            }

            Synchronize();
            DecrementCacheabble(_inputsSecondary);
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, uint batchSize)
        {
            _inputDimensions = inputShapes.Length;
            _outputDimensions = _inputDimensions + _secondaryDimensions;

            _outputShapes = new Shape[_outputDimensions];

            _layerInfos = new ILayerInfo[_inputDimensions + _secondaryDimensions];
            for (int i = 0; i < _inputDimensions; i++)
            {
                StaticLayerInfo layer = new StaticLayerInfo()
                {
                    Width = inputShapes[i].Width,
                    Length = inputShapes[i].Length,
                };

                _layerInfos[i] = layer;

                _outputShapes[i] = new Shape(layer.Width, layer.Length);
            }

            for (int i = 0; i < _secondaryDimensions; i++)
            {
                StaticLayerInfo layer = new StaticLayerInfo()
                {
                    Width = _inputsSecondary[i, 0].Width,
                    Length = _inputsSecondary[i, 0].Length
                };
                _layerInfos[i + _inputDimensions] = layer;
                _outputShapes[_inputDimensions + i] = new Shape(layer.Width, layer.Length);
            }

            _buffers = buffers;
            for (int i = 0; i < _outputDimensions; i++)
                buffers.OutputDimensionArea(i, _outputShapes[i].Area);

            return _outputShapes;
        }*/
        public override string Name => throw new NotImplementedException();

        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            throw new NotImplementedException();
        }

        public override void Forward()
        {
            throw new NotImplementedException();
        }

        public override void Reset()
        {
            throw new NotImplementedException();
        }

        public override Shape Startup(Shape inputShapes, IOBuffers buffers, int batchSize)
        {
            throw new NotImplementedException();
        }
    }
}