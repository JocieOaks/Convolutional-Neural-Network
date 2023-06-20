using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="FullyConnected"/> class is a <see cref="Layer"/> for that connects every input node to every output node,
    /// so that every output <see cref="FeatureMap"/> for an image is based on all every input <see cref="FeatureMap"/>.
    /// </summary>
    [Serializable]
    public class FullyConnected : Layer, IPrimaryLayer
    {
        private MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense>[] _deviceInfos;
        private MemoryBuffer1D<float, Stride1D.Dense>[,] _deviceMultiplierGradients;
        private MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceMultipliers;
        private new MemoryBuffer1D<float, Stride1D.Dense>[,] _deviceOutputs;

        [JsonProperty] private FeatureMap _matrixBlue;
        [JsonProperty] private FeatureMap _matrixGreen;
        [JsonProperty] private FeatureMap _matrixRed;

        private int _dimensionMultiplier;

        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureMap"/> class.
        /// </summary>
        public FullyConnected() : base(1, 1)
        {
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
        public override string Name => "Fully Connected Layer";

        /// <summary>
        /// Backpropagates through the layer without updating any of the filter weights. Called when learning rate is zero.
        /// </summary>
        private void BackwardsNoUpdate()
        {
            Context context = ConvolutionalNeuralNetwork.Utility.Context;
            Accelerator accelerator = ConvolutionalNeuralNetwork.Utility.Accelerator;

            var backwardsOutKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(BackwardsOutKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceOutGradients[i, j] = _outGradients[i, j].AllocateFloat(accelerator);
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);
                }
            }

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);
                for (int j = 0; j < _outputDimensions; j++)
                {
                    _deviceMultipliers[i, j] = accelerator.Allocate1D(new Color[] { _matrixRed[i, j], _matrixGreen[i, j], _matrixBlue[i, j] });
                    for (int k = 0; k < _batchSize; k++)
                    {
                        backwardsOutKernal(index, _deviceInGradients[j, k].View, _deviceMultipliers[i, j].View, _deviceOutGradients[i, k].View, _deviceInfos[i].View);
                    }
                }
            }

            accelerator.Synchronize();

            for (int j = 0; j < _batchSize; j++)
            {
                for (int i = 0; i < _inputDimensions; i++)
                {
                    _outGradients[i, j].CopyFromBuffer(_deviceOutGradients[i, j]);
                    _deviceOutGradients[i, j].Dispose();
                }
                for (int i = 0; i < _outputDimensions; i++)
                {
                    _deviceInGradients[i, j].Dispose();
                }
            }

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _outputDimensions; j++)
                {
                    _deviceMultipliers[i, j].Dispose();
                }

                _deviceInfos[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public override void Backwards(float learningRate)
        {
            if (learningRate == 0)
                BackwardsNoUpdate();
            else
                BackwardsUpdate(learningRate);
        }

        /// Perform standard backpropagation through the layer, updating it's weights. Called when learning rate is greater than 0.
        /// </summary>
        /// <param name="learningRate">Controls how much the layer is updated with each backpropagation.</param>
        private void BackwardsUpdate(float learningRate)
        {
            Context context = ConvolutionalNeuralNetwork.Utility.Context;
            Accelerator accelerator = ConvolutionalNeuralNetwork.Utility.Accelerator;

            var backwardsOutKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(BackwardsOutKernal);

            var backwardsGradientKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(BackwardsGradientKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceOutGradients[i, j] = _outGradients[i, j].AllocateFloat(accelerator);
                    _deviceInputs[i, j] = _inputs[i, j].Allocate(accelerator);
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);
                }
            }

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);
                for (int j = 0; j < _outputDimensions; j++)
                {
                    _deviceMultipliers[i, j] = accelerator.Allocate1D(new Color[] { _matrixRed[i, j], _matrixGreen[i, j], _matrixBlue[i, j] });
                    _deviceMultiplierGradients[i, j] = accelerator.Allocate1D<float>(9);
                    for (int k = 0; k < _batchSize; k++)
                    {
                        backwardsOutKernal(index, _deviceInGradients[j, k].View, _deviceMultipliers[i, j].View, _deviceOutGradients[i, k].View, _deviceInfos[i].View);
                        backwardsGradientKernal(index, _deviceInGradients[j, k].View, _deviceInputs[i, k].View, _deviceMultiplierGradients[i, j].View, _deviceInfos[i].View);
                    }
                }
            }

            accelerator.Synchronize();

            for (int j = 0; j < _batchSize; j++)
            {
                for (int i = 0; i < _inputDimensions; i++)
                {
                    _outGradients[i, j].CopyFromBuffer(_deviceOutGradients[i, j]);

                    _deviceOutGradients[i, j].Dispose();
                    _deviceInputs[i, j].Dispose();
                }
                for (int i = 0; i < _outputDimensions; i++)
                {
                    _deviceInGradients[i, j].Dispose();
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

                    _matrixRed[i, j] -= new Color(multiplierGradients[0], multiplierGradients[1], multiplierGradients[2]).Clamp(1) * learningRate;
                    _matrixGreen[i, j] -= new Color(multiplierGradients[3], multiplierGradients[4], multiplierGradients[5]).Clamp(1) * learningRate;
                    _matrixBlue[i, j] -= new Color(multiplierGradients[6], multiplierGradients[7], multiplierGradients[8]).Clamp(1) * learningRate;
                }

                _deviceInfos[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            Accelerator accelerator = ConvolutionalNeuralNetwork.Utility.Accelerator;

            var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<Color>, ArrayView<SingleLayerInfo>>(ForwardKernal);

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceOutputs[i, j] = _outputs[i, j].AllocateFloat(accelerator);
                }
            }

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });

                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInputs[i, j] = _inputs[i, j].Allocate(accelerator);
                }
            }

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index3D index = new(Infos(i).Width, Infos(i).Length, 3);
                for (int j = 0; j < _outputDimensions; j++)
                {
                    _deviceMultipliers[i, j] = accelerator.Allocate1D(new Color[] { _matrixRed[i, j], _matrixGreen[i, j], _matrixBlue[i, j] });

                    for (int k = 0; k < _batchSize; k++)
                    {
                        forwardKernal(index, _deviceInputs[i, k].View, _deviceOutputs[j, k].View, _deviceMultipliers[i, j].View, _deviceInfos[i].View);
                    }
                }
            }

            accelerator.Synchronize();

            for (int j = 0; j < _batchSize; j++)
            {
                for (int i = 0; i < _outputDimensions; i++)
                {
                    _outputs[i, j].CopyFromBuffer(_deviceOutputs[i, j]);

                    _deviceOutputs[i, j].Dispose();
                }

                for (int i = 0; i < _inputDimensions; i++)
                {
                    _deviceInputs[i, j].Dispose();
                }
            }

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _outputDimensions; j++)
                {
                    _deviceMultipliers[i, j].Dispose();
                }
                _deviceInfos[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public override void Reset()
        {
            float variance = 0.666f / (_inputDimensions + _outputDimensions);
            float stdDev = MathF.Sqrt(variance);
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _outputDimensions; j++)
                {
                    _matrixRed[i, j] = Color.RandomGauss(0, stdDev);
                    _matrixGreen[i, j] = Color.RandomGauss(0, stdDev);
                    _matrixBlue[i, j] = Color.RandomGauss(0, stdDev);
                }
            }
        }

        /// <inheritdoc/>
        public override (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] inputs, FeatureMap[,] outGradients)
        {
            if (_matrixRed == null)
            {
                if (_outputDimensions != 0)
                    BaseStartup(inputs, outGradients, -inputs.GetLength(0) / _outputDimensions);
                else if (_dimensionMultiplier != 0)
                    BaseStartup(inputs, outGradients, _dimensionMultiplier);
                else
                    BaseStartup(inputs, outGradients, 1);

                float variance = 0.666f / (_inputDimensions + _outputDimensions);
                float stdDev = MathF.Sqrt(variance);
                _matrixRed = new FeatureMap(_inputDimensions, _outputDimensions);
                _matrixGreen = new FeatureMap(_inputDimensions, _outputDimensions);
                _matrixBlue = new FeatureMap(_inputDimensions, _outputDimensions);
                for (int i = 0; i < _inputDimensions; i++)
                {
                    for (int j = 0; j < _outputDimensions; j++)
                    {
                        _matrixRed[i, j] = Color.RandomGauss(0, stdDev);
                        _matrixGreen[i, j] = Color.RandomGauss(0, stdDev);
                        _matrixBlue[i, j] = Color.RandomGauss(0, stdDev);
                    }
                }
            }
            else
            {
                BaseStartup(inputs, outGradients, -_matrixRed.Width / _matrixRed.Length);
            }
            _deviceInfos = new MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense>[_inputDimensions];
            _deviceMultiplierGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions, _outputDimensions];
            _deviceMultipliers = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _outputDimensions];
            _deviceOutputs = new MemoryBuffer1D<float, Stride1D.Dense>[_outputDimensions, _batchSize];
            return (_outputs, _inGradients);
        }

        private static void BackwardsGradientKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<Color> input, ArrayView<float> multiplierGradient, ArrayView<SingleLayerInfo> info)
        {
            int mapsIndex = info[0].Index(index.X, index.Y);
            for (int i = 0; i < 3; i++)
            {
                Atomic.Add(ref multiplierGradient[index.Z * 3 + i], inGradient[mapsIndex][index.Z] * input[mapsIndex][i]);
            }
        }

        private static void BackwardsOutKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<Color> multiplier, ArrayView<float> outGradient, ArrayView<SingleLayerInfo> info)
        {
            int mapsIndex = info[0].Index(index.X, index.Y);
            float transposeDot = 0;
            for (int i = 0; i < 3; i++)
            {
                transposeDot += inGradient[mapsIndex][i] * multiplier[i][index.Z];
            }
            Atomic.Add(ref outGradient[mapsIndex * 3 + index.Z], transposeDot);
        }

        private static void ForwardKernal(Index3D index, ArrayView<Color> input, ArrayView<float> output, ArrayView<Color> multiplier, ArrayView<SingleLayerInfo> info)
        {
            int mapsIndex = info[0].Index(index.X, index.Y);
            Atomic.Add(ref output[mapsIndex * 3 + index.Z], Color.Dot(input[mapsIndex], multiplier[index.Z]));
        }

        /// <summary>
        /// Gets the <see cref="SingleLayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="SingleLayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="SingleLayerInfo"/> corresponding to an input dimension.</returns>
        private SingleLayerInfo Infos(int index)
        {
            return (SingleLayerInfo)_layerInfos[index];
        }
    }
}