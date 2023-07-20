using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="SkipConnectionSplit"/> class is a <see cref="Layer"/> that creates two sets of the same <see cref="FeatureMap"/>s, sending
    /// one as input to the next <see cref="Layer"/> and sending one to a <see cref="SkipConnectionConcatenate"/> later in the <see cref="Network"/>.
    /// </summary>
    public class SkipConnectionSplit : Layer, IStructuralLayer
    {
        /*
        /// <inheritdoc/>
        public override string Name => "Skip Connection Layer";

        private FeatureMap[,] _inGradientSecondary;
        private SkipConnectionConcatenate _concatenationLayer;
        private FeatureMap[,] _outputs;

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

        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> s_backwardsAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(BackwardsKernel);

        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index1D index = new(Infos(i).Area);
                for (int j = 0; j < _batchSize; j++)
                {
                    s_backwardsAction(index, _buffers.InGradient[i, j], _inGradientSecondary[i, j].GetArrayView<float>(), _buffers.OutGradient[i, j]);
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
                    GPU.GPUManager.CopyAction(index, _buffers.Input[i, j], _outputs[i, j].GetArrayViewEmpty<float>());
                    GPU.GPUManager.CopyAction(index, _buffers.Input[i, j], _buffers.Output[i, j]);
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
        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, uint batchSize)
        {
            _outputDimensions = _inputDimensions = inputShapes.GetLength(0);
            _buffers = buffers;

            _batchSize = (uint)inputShapes.GetLength(1);
            _layerInfos = new ILayerInfo[_inputDimensions];
            for (int i = 0; i < _inputDimensions; i++)
            {
                _layerInfos[i] = new StaticLayerInfo()
                {
                    Width = inputShapes[i].Width,
                    Length = inputShapes[i].Length,
                };
            }

            _outputShapes = inputShapes;

            _outputs = new FeatureMap[_inputDimensions, batchSize];
            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < batchSize; j++)
                {
                    _outputs[i, j] = new FeatureMap(_outputShapes[i]);
                }
            }

            _inGradientSecondary = new FeatureMap[_outputDimensions, _batchSize];

            _concatenationLayer.Connect(_outputs, _inGradientSecondary);

            return inputShapes;
        }

        private static void BackwardsKernel(Index1D index, ArrayView<float> inGradient1, ArrayView<float> inGradient2, ArrayView<float> outGradient)
        {
            outGradient[index.X] = inGradient1[index.X] + inGradient2[index.X];
        }

        /// <summary>
        /// Gets the <see cref="StaticLayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="StaticLayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="StaticLayerInfo"/> corresponding to an input dimension.</returns>
        private StaticLayerInfo Infos(int index)
        {
            return (StaticLayerInfo)_layerInfos[index];
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

        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, int batchSize)
        {
            throw new NotImplementedException();
        }
    }
}