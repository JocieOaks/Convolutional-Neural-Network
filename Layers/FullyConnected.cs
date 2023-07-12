using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="FullyConnected"/> class is a <see cref="Layer"/> for that connects every input node to every output node,
    /// so that every output <see cref="FeatureMap"/> for an image is based on all every input <see cref="FeatureMap"/>.
    /// </summary>
    [Serializable]
    public class FullyConnected : Layer, IPrimaryLayer
    {
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<Color>, ArrayView<float>, ArrayView<StaticLayerInfo>> s_backwardsGradientAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<Color>, ArrayView<float>, ArrayView<StaticLayerInfo>>(BackwardsGradientKernal);
        private static readonly Action<Index3D, ArrayView<float>, ArrayView<Color>, ArrayView<float>, ArrayView<StaticLayerInfo>> s_backwardsOutAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<Color>, ArrayView<float>, ArrayView<StaticLayerInfo>>(BackwardsOutKernal);
        private static readonly Action<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<Color>, ArrayView<StaticLayerInfo>> s_forwardAction = Utility.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<Color>, ArrayView<StaticLayerInfo>>(ForwardKernal);
        private MemoryBuffer1D<StaticLayerInfo, Stride1D.Dense>[] _deviceInfos;
        private MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceInputs;
        private int _dimensionMultiplier;
        private FeatureMap[,] _inputs;
        [JsonProperty] private Weights _filter;

        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureMap"/> class.
        /// </summary>
        public FullyConnected() : base(1, 1)
        {
        }

        /// <inheritdoc/>
        public override string Name => "Fully Connected Layer";

        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            if (learningRate == 0)
                BackwardsNoUpdate();
            else
                BackwardsUpdate(learningRate, firstMomentDecay, secondMomentDecay);
        }
        /// <inheritdoc/>
        public override void Forward()
        {
            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _buffers.OutputsColor[i, j].SubView(0, Infos(i).Area).MemSetToZero();
                }
            }

            ArrayView<Color> deviceFilter = _filter.FilterGPU();

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);
                for (int j = 0; j < _outputDimensions; j++)
                {
                    for (int k = 0; k < _batchSize; k++)
                    {
                        s_forwardAction(index, _buffers.InputsColor[i, k], _buffers.OutputsFloat[j, k], deviceFilter.SubView(MultiplierIndex(i, j), 3), _deviceInfos[i].View);
                    }
                }
            }

            Utility.Accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _inputs[i, j].CopyFromBuffer(_buffers.InputsColor[i, j]);
                }
            }
            _filter.DisposeFilter();
        }

        private int MultiplierIndex(int i, int j)
        {
            return 3 * (i * _outputDimensions + j);
        }

        /// <summary>
        /// Called when the layer is deserialized.
        /// Temporary function to allow for loading models that were created before Adam optimization was used implemented.
        /// </summary>
        /// <param name="context">The streaming context for deserialization.</param>
        [OnDeserialized]
        public void OnDeserialized(StreamingContext context)
        {
        }

        /// <inheritdoc/>
        public override void Reset()
        {
            float variance = 0.666f / (_inputDimensions + _outputDimensions);
            float stdDev = MathF.Sqrt(variance);
            _filter.Reset(0, stdDev);
        }

        /// <summary>
        /// Sets the number of dimensions for the output. The input dimensions will need to be a multiple of the output dimensions
        /// or vice versa. Overwrites <see cref="SetOutputMultiplier(int)"/>.
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
        /// <param name="multiplier">The factor to multiply the input dimensions to set the output dimensions.</param>
        public void SetOutputMultiplier(int multiplier)
        {
            _dimensionMultiplier = multiplier;
        }

        /// <inheritdoc/>
        public override FeatureMap[,] Startup(FeatureMap[,] inputs, IOBuffers buffers)
        {
            if (_filter == null)
            {
                if (_outputDimensions != 0)
                    BaseStartup(inputs, buffers, -inputs.GetLength(0) / _outputDimensions);
                else if (_dimensionMultiplier != 0)
                    BaseStartup(inputs, buffers, _dimensionMultiplier);
                else
                    BaseStartup(inputs, buffers, 1);

                float variance = 0.666f / (_inputDimensions + _outputDimensions);
                float stdDev = MathF.Sqrt(variance);

                _filter = new Weights(3 * _inputDimensions * _outputDimensions, 0, stdDev);
            }
            else
            {
                _inputDimensions = inputs.GetLength(0);
                BaseStartup(inputs, buffers, -3 * _inputDimensions * _inputDimensions / _filter.Length);
            }
            _inputs = inputs;

            _deviceInfos = new MemoryBuffer1D<StaticLayerInfo, Stride1D.Dense>[_inputDimensions];
            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = Utility.Accelerator.Allocate1D(new StaticLayerInfo[] { Infos(i) });
            }
            _deviceInputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];
            return _outputs;
        }
        private static void BackwardsGradientKernal(Index3D index, ArrayView<float> inGradient, ArrayView<Color> input, ArrayView<float> multiplierGradient, ArrayView<StaticLayerInfo> info)
        {
            int mapsIndex = info[0].Index(index.X, index.Y);
            for (int i = 0; i < 3; i++)
            {
                Atomic.Add(ref multiplierGradient[index.Z * 3 + i], inGradient[3 * mapsIndex + index.Z] * input[mapsIndex][i]);
            }
        }

        private static void BackwardsOutKernal(Index3D index, ArrayView<float> inGradient, ArrayView<Color> multiplier, ArrayView<float> outGradient, ArrayView<StaticLayerInfo> info)
        {
            int mapsIndex = info[0].Index(index.X, index.Y);
            float transposeDot = 0;
            for (int i = 0; i < 3; i++)
            {
                transposeDot += inGradient[3 * mapsIndex + i] * multiplier[i][index.Z];
            }
            Atomic.Add(ref outGradient[mapsIndex * 3 + index.Z], transposeDot);
        }

        private static void ForwardKernal(Index3D index, ArrayView<Color> input, ArrayView<float> output, ArrayView<Color> multiplier, ArrayView<StaticLayerInfo> info)
        {
            int mapsIndex = info[0].Index(index.X, index.Y);
            Atomic.Add(ref output[mapsIndex * 3 + index.Z], Color.Dot(input[mapsIndex], multiplier[index.Z]));
        }

        /// <summary>
        /// Backpropagates through the layer without updating any of the filter weights. Called when learning rate is zero.
        /// </summary>
        private void BackwardsNoUpdate()
        {

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _buffers.OutGradientsColor[i, j].SubView(0, Infos(i).Area).MemSetToZero();
                }
            }

            ArrayView<Color> deviceFilter = _filter.FilterGPU();

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);
                for (int j = 0; j < _outputDimensions; j++)
                {
                    int multIndex = MultiplierIndex(i, j);
                    for (int k = 0; k < _batchSize; k++)
                    {
                        s_backwardsOutAction(index, _buffers.InGradientsFloat[j, k], deviceFilter.SubView(multIndex, 3), _buffers.OutGradientsFloat[i, k], _deviceInfos[i].View);
                    }
                }
            }

            Utility.Accelerator.Synchronize();

            _filter.DisposeFilter();
        }

        /// Perform standard backpropagation through the layer, updating it's weights. Called when learning rate is greater than 0.
        /// </summary>
        /// <param name="learningRate">The overall learning rate for the layer updates, corrected for the influence of bias in the first and second moments.</param>
        /// <param name="firstMomentDecay">The exponential decay rate for the first moment.</param>
        /// <param name="secondMomentDecay">The exponential decay rate for the second moment.</param>
        private void BackwardsUpdate(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInputs[i, j] = _inputs[i, j].Allocate(Utility.Accelerator);
                }
            }

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _buffers.OutGradientsColor[i, j].SubView(0, Infos(i).Area).MemSetToZero();
                }
            }

            var deviceFilter = _filter.FilterGPU();
            var deviceGradient = _filter.GradientGPU();

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);
                for (int j = 0; j < _outputDimensions; j++)
                {
                    int multIndex = MultiplierIndex(i, j);
                    for (int k = 0; k < _batchSize; k++)
                    {
                        s_backwardsOutAction(index, _buffers.InGradientsFloat[j, k], deviceFilter.SubView(multIndex, 3), _buffers.OutGradientsFloat[i, k], _deviceInfos[i].View);
                        s_backwardsGradientAction(index, _buffers.InGradientsFloat[j, k], _deviceInputs[i, k].View, deviceGradient.SubView(3 * multIndex, 9), _deviceInfos[i].View);
                    }
                }
            }

            Utility.Accelerator.Synchronize();
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInputs[i, j].Dispose();
                }
            }

            float[] multiplierGradients = new float[9];

            _filter.DisposeFilter();
            _filter.DisposeGradient();
            _filter.UpdateFilter(learningRate, firstMomentDecay, secondMomentDecay);
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

        public void FilterTest(int outputMultiplier)
        {
            FeatureMap[,] input = FilterTestSetup(outputMultiplier);

            _filter.TestFilterGradient(this, input, _outputs[0, 0], 0, _buffers);
        }
    }
}