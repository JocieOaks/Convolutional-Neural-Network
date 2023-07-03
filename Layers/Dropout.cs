using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="Dropout"/> class is a <see cref="Layer"/> that adds noise to a <see cref="FeatureMap"/> by randomly
    /// dropping values from the <see cref="FeatureMap"/>.
    /// </summary>
    public class Dropout : Layer, ISecondaryLayer
    {
        private MemoryBuffer1D<int, Stride1D.Dense>[] _deviceDropout;
        private int[][] _dropout;
        [JsonProperty] private float _dropoutRate = 0.2f;

        /// <summary>
        /// Initializes a new instance of the <see cref="Dropout"/> class.
        /// </summary>
        /// <param name="dropoutRate">The frequency in which values are dropped.</param>
        public Dropout(float dropoutRate) : base(1, 1)
        {
            _dropoutRate = dropoutRate;
        }

        /// <summary>
        /// A default constructor to be used when deserializing.
        /// </summary>
        [JsonConstructor]
        private Dropout() : base(1, 1)
        {
        }

        /// <inheritdoc/>
        public override string Name => "Dropout Layer";

        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            Accelerator accelerator = Utility.Accelerator;
            var backwardsKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<int>, ArrayView<float>>(BackwardsKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index2D index = new(Infos(i).Area, 3);
                _deviceDropout[i] = accelerator.Allocate1D(_dropout[i]);
                for (int j = 0; j < _batchSize; j++)
                {
                    backwardsKernal(index, _buffers.InGradientsFloat[i, j], _deviceDropout[i].View, _buffers.OutGradientsFloat[i, j]);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceDropout[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            Context context = ConvolutionalNeuralNetwork.Utility.Context;
            Accelerator accelerator = ConvolutionalNeuralNetwork.Utility.Accelerator;
            var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Color>, ArrayView<int>, ArrayView<Color>>(ForwardKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _dropout[i].Length; j++)
                {
                    _dropout[i][j] = Utility.Random.NextDouble() < _dropoutRate ? 0 : 1;
                }
                Index1D index = new(Infos(i).Area);
                _deviceDropout[i] = accelerator.Allocate1D(_dropout[i]);
                for (int j = 0; j < _batchSize; j++)
                {
                    forwardKernal(index, _buffers.InputsColor[i, j], _deviceDropout[i].View, _buffers.OutputsColor[i, j]);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceDropout[i].Dispose();
            }
        }

        /// <summary>
        /// Forward propogates through the <see cref="Dropout"/> layer, but instead of randomly dropping values, it just adjusts
        /// every <see cref="Color"/> based on it's dropout rate so that the output is scaled similar to while training.
        /// </summary>
        public void ForwardInference()
        {
            Context context = ConvolutionalNeuralNetwork.Utility.Context;
            Accelerator accelerator = ConvolutionalNeuralNetwork.Utility.Accelerator;
            var inferenceKernal = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Color>, ArrayView<float>, ArrayView<Color>>(InferenceKernal);

            MemoryBuffer1D<float, Stride1D.Dense> deviceRatio = accelerator.Allocate1D(new float[] { 1 - _dropoutRate });
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index1D index = new(Infos(i).Area);
                for (int j = 0; j < _batchSize; j++)
                {
                    inferenceKernal(index, _buffers.InputsColor[i, j], deviceRatio.View, _buffers.OutputsColor[i, j]);
                }
            }

            accelerator.Synchronize();
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override FeatureMap[,] Startup(FeatureMap[,] input, IOBuffers buffers)
        {
            BaseStartup(input, buffers);

            _dropout = new int[_inputDimensions][];
            for (int i = 0; i < _inputDimensions; i++)
            {
                _dropout[i] = new int[input[i, 0].Area * 3];
            }
            _deviceDropout = new MemoryBuffer1D<int, Stride1D.Dense>[_inputDimensions];

            return _outputs;
        }

        private static void BackwardsKernal(Index2D index, ArrayView<float> inGradient, ArrayView<int> dropout, ArrayView<float> outGradient)
        {
            int arrayIndex = index.X * 3 + index.Y;
            outGradient[arrayIndex] = dropout[arrayIndex] == 0 ? 0 : inGradient[arrayIndex];
        }

        private static void ForwardKernal(Index1D index, ArrayView<Color> input, ArrayView<int> dropout, ArrayView<Color> output)
        {
            float r = dropout[3 * index] == 0 ? 0 : input[index].R;
            float g = dropout[3 * index + 1] == 0 ? 0 : input[index].G;
            float b = dropout[3 * index + 2] == 0 ? 0 : input[index].B;
            output[index] = new Color(r, g, b);
        }

        private static void InferenceKernal(Index1D index, ArrayView<Color> input, ArrayView<float> dropoutRate, ArrayView<Color> output)
        {
            output[index] = input[index] * dropoutRate[0];
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