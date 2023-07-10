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
        [JsonProperty] private ColorTensor _blueFirstMoment;
        [JsonProperty] private ColorTensor _blueSecondMoment;
        private MemoryBuffer1D<StaticLayerInfo, Stride1D.Dense>[] _deviceInfos;
        private MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceInputs;
        private MemoryBuffer1D<float, Stride1D.Dense>[,] _deviceMultiplierGradients;
        private MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceMultipliers;
        private int _dimensionMultiplier;
        [JsonProperty] private ColorTensor _greenFirstMoment;
        [JsonProperty] private ColorTensor _greenSecondMoment;
        private FeatureMap[,] _inputs;
        [JsonProperty] private ColorTensor _matrixBlue;
        [JsonProperty] private ColorTensor _matrixGreen;
        [JsonProperty] private ColorTensor _matrixRed;
        [JsonProperty] private ColorTensor _redFirstMoment;
        [JsonProperty] private ColorTensor _redSecondMoment;

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

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);
                for (int j = 0; j < _outputDimensions; j++)
                {
                    _deviceMultipliers[i, j] = Utility.Accelerator.Allocate1D(new Color[] { _matrixRed[i, j], _matrixGreen[i, j], _matrixBlue[i, j] });

                    for (int k = 0; k < _batchSize; k++)
                    {
                        s_forwardAction(index, _buffers.InputsColor[i, k], _buffers.OutputsFloat[j, k], _deviceMultipliers[i, j].View, _deviceInfos[i].View);
                    }
                }
            }

            Utility.Accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _outputDimensions; j++)
                {
                    _deviceMultipliers[i, j].Dispose();
                }

                for (int j = 0; j < _batchSize; j++)
                {
                    _inputs[i, j].CopyFromBuffer(_buffers.InputsColor[i, j]);
                }
            }
        }

        /// <summary>
        /// Called when the layer is deserialized.
        /// Temporary function to allow for loading models that were created before Adam optimization was used implemented.
        /// </summary>
        /// <param name="context">The streaming context for deserialization.</param>
        [OnDeserialized]
        public void OnDeserialized(StreamingContext context)
        {
            if (_blueFirstMoment == null || _blueSecondMoment == null || _greenFirstMoment == null || _greenSecondMoment == null || _redFirstMoment == null || _redSecondMoment == null)
            {
                _blueFirstMoment = new ColorTensor(_matrixBlue.Width, _matrixBlue.Length);
                _blueSecondMoment = new ColorTensor(_matrixBlue.Width, _matrixBlue.Length);
                _greenFirstMoment = new ColorTensor(_matrixGreen.Width, _matrixGreen.Length);
                _greenSecondMoment = new ColorTensor(_matrixGreen.Width, _matrixGreen.Length);
                _redFirstMoment = new ColorTensor(_matrixRed.Width, _matrixRed.Length);
                _redSecondMoment = new ColorTensor(_matrixRed.Width, _matrixRed.Length);
            }
        }

        /// <inheritdoc/>
        public override void Reset()
        {
            float variance = 0.666f / (_inputDimensions + _outputDimensions);
            float stdDev = MathF.Sqrt(variance);
            _matrixRed = ColorTensor.Random(_inputDimensions, _outputDimensions, 0, stdDev);
            _matrixGreen = ColorTensor.Random(_inputDimensions, _outputDimensions, 0, stdDev);
            _matrixBlue = ColorTensor.Random(_inputDimensions, _outputDimensions, 0, stdDev);

            _blueFirstMoment = new ColorTensor(_inputDimensions, _outputDimensions);
            _blueSecondMoment = new ColorTensor(_inputDimensions, _outputDimensions);
            _greenFirstMoment = new ColorTensor(_inputDimensions, _outputDimensions);

            _greenSecondMoment = new ColorTensor(_inputDimensions, _outputDimensions);
            _redFirstMoment = new ColorTensor(_inputDimensions, _outputDimensions);
            _redSecondMoment = new ColorTensor(_inputDimensions, _outputDimensions);
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
            if (_matrixRed == null)
            {
                if (_outputDimensions != 0)
                    BaseStartup(inputs, buffers, -inputs.GetLength(0) / _outputDimensions);
                else if (_dimensionMultiplier != 0)
                    BaseStartup(inputs, buffers, _dimensionMultiplier);
                else
                    BaseStartup(inputs, buffers, 1);

                float variance = 0.666f / (_inputDimensions + _outputDimensions);
                float stdDev = MathF.Sqrt(variance);

                _matrixRed = ColorTensor.Random(_inputDimensions, _outputDimensions, 0, stdDev);
                _matrixGreen = ColorTensor.Random(_inputDimensions, _outputDimensions, 0, stdDev);
                _matrixBlue = ColorTensor.Random(_inputDimensions, _outputDimensions, 0, stdDev);

                _blueFirstMoment = new ColorTensor(_inputDimensions, _outputDimensions);
                _blueSecondMoment = new ColorTensor(_inputDimensions, _outputDimensions);
                _greenFirstMoment = new ColorTensor(_inputDimensions, _outputDimensions);

                _greenSecondMoment = new ColorTensor(_inputDimensions, _outputDimensions);
                _redFirstMoment = new ColorTensor(_inputDimensions, _outputDimensions);
                _redSecondMoment = new ColorTensor(_inputDimensions, _outputDimensions);
            }
            else
            {
                BaseStartup(inputs, buffers, -_matrixRed.Width / _matrixRed.Length);
            }
            _inputs = inputs;

            _deviceInfos = new MemoryBuffer1D<StaticLayerInfo, Stride1D.Dense>[_inputDimensions];
            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = Utility.Accelerator.Allocate1D(new StaticLayerInfo[] { Infos(i) });
            }
            _deviceMultiplierGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions, _outputDimensions];
            _deviceMultipliers = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _outputDimensions];
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

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);
                for (int j = 0; j < _outputDimensions; j++)
                {
                    _deviceMultipliers[i, j] = Utility.Accelerator.Allocate1D(new Color[] { _matrixRed[i, j], _matrixGreen[i, j], _matrixBlue[i, j] });
                    for (int k = 0; k < _batchSize; k++)
                    {
                        s_backwardsOutAction(index, _buffers.InGradientsFloat[j, k], _deviceMultipliers[i, j].View, _buffers.OutGradientsFloat[i, k], _deviceInfos[i].View);
                    }
                }
            }

            Utility.Accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _outputDimensions; j++)
                {
                    _deviceMultipliers[i, j].Dispose();
                }
            }
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

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);
                for (int j = 0; j < _outputDimensions; j++)
                {
                    _deviceMultipliers[i, j] = Utility.Accelerator.Allocate1D(new Color[] { _matrixRed[i, j], _matrixGreen[i, j], _matrixBlue[i, j] });
                    _deviceMultiplierGradients[i, j] = Utility.Accelerator.Allocate1D<float>(9);
                    _deviceMultiplierGradients[i, j].MemSetToZero();
                    for (int k = 0; k < _batchSize; k++)
                    {
                        s_backwardsOutAction(index, _buffers.InGradientsFloat[j, k], _deviceMultipliers[i, j].View, _buffers.OutGradientsFloat[i, k], _deviceInfos[i].View);
                        s_backwardsGradientAction(index, _buffers.InGradientsFloat[j, k], _deviceInputs[i, k].View, _deviceMultiplierGradients[i, j].View, _deviceInfos[i].View);
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

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _outputDimensions; j++)
                {
                    _deviceMultiplierGradients[i, j].CopyToCPU(multiplierGradients);
                    _deviceMultiplierGradients[i, j].Dispose();
                    _deviceMultipliers[i, j].Dispose();

                    UpdateWeight(learningRate, firstMomentDecay, secondMomentDecay, i, j, multiplierGradients);
                }
            }
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

        /// <summary>
        /// Updates weights along with the first and second moments.
        /// </summary>
        /// <param name="learningRate">The overall learning rate for the layer updates, corrected for the influence of bias in the first and second moments.</param>
        /// <param name="firstMomentDecay">The exponential decay rate for the first moment.</param>
        /// <param name="secondMomentDecay">The exponential decay rate for the second moment.</param>
        /// <param name="index1">The first dimension index of weights being updated.</param>
        /// <param name="index2">The second dimension index of the weights being updated.</param>
        /// <param name="multiplierGradients">The gradients of the weights.</param>
        private void UpdateWeight(float learningRate, float firstMomentDecay, float secondMomentDecay, int index1, int index2, float[] multiplierGradients)
        {
            Color gradient = new Color(multiplierGradients[0], multiplierGradients[1], multiplierGradients[2]);
            Color first = _redFirstMoment[index1, index2] = firstMomentDecay * _redFirstMoment[index1, index2] + (1 - firstMomentDecay) * gradient;
            Color second = _redSecondMoment[index1, index2] = secondMomentDecay * _redSecondMoment[index1, index2] + (1 - secondMomentDecay) * Color.Pow(gradient, 2);
            _matrixRed[index1, index2] -= learningRate * first / (Color.Pow(second, 0.5f) + Utility.AsymptoteErrorColor);


            gradient = new Color(multiplierGradients[3], multiplierGradients[4], multiplierGradients[5]);
            first = _greenFirstMoment[index1, index2] = firstMomentDecay * _greenFirstMoment[index1, index2] + (1 - firstMomentDecay) * gradient;
            second = _greenSecondMoment[index1, index2] = secondMomentDecay * _greenSecondMoment[index1, index2] + (1 - secondMomentDecay) * Color.Pow(gradient, 2);
            _matrixGreen[index1, index2] -= learningRate * first / (Color.Pow(second, 0.5f) + Utility.AsymptoteErrorColor);

            gradient = new Color(multiplierGradients[6], multiplierGradients[7], multiplierGradients[8]);
            first = _blueFirstMoment[index1, index2] = firstMomentDecay * _blueFirstMoment[index1, index2] + (1 - firstMomentDecay) * gradient;
            second = _blueSecondMoment[index1, index2] = secondMomentDecay * _blueSecondMoment[index1, index2] + (1 - secondMomentDecay) * Color.Pow(gradient, 2);
            _matrixBlue[index1, index2] -= learningRate * first / (Color.Pow(second, 0.5f) + Utility.AsymptoteErrorColor);

        }
    }
}