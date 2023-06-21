using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="Scaling"/> class is a <see cref="Layer"/> for upsampling or downsampling a <see cref="FeatureMap"/>.
    /// </summary>
    public class Scaling : Layer, IStructuralLayer
    {
        private MemoryBuffer1D<ScalingLayerInfo, Stride1D.Dense>[] _deviceInfos;
        [JsonProperty] private int _outputLength;
        [JsonProperty] private int _outputWidth;
        private float _scaleLength;
        private float _scaleWidth;
        /// <inheritdoc/>
        public override string Name => "Scaling Layer";

        /// <inheritdoc/>
        public override void Backwards(float learningRatee, float firstMomentDecay, float secondMomentDecay)
        {
            Context context = ConvolutionalNeuralNetwork.Utility.Context;
            Accelerator accelerator = ConvolutionalNeuralNetwork.Utility.Accelerator;

            var backwardsKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<ScalingLayerInfo>>(BackwardsKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = accelerator.Allocate1D(new ScalingLayerInfo[] { Infos(i) });
                Index3D index = new(Infos(i).OutputWidth, Infos(i).OutputLength, 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);
                    _deviceOutGradients[i, j] = _outGradients[i, j].AllocateFloat(accelerator);

                    backwardsKernal(index, _deviceInGradients[i, j].View, _deviceOutGradients[i, j].View, _deviceInfos[i].View);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _outGradients[i, j].CopyFromBuffer(_deviceOutGradients[i, j]);
                    _deviceOutGradients[i, j].Dispose();
                    _deviceInputs[i, j].Dispose();
                    _deviceInGradients[i, j].Dispose();
                }
                _deviceInfos[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            Context context = ConvolutionalNeuralNetwork.Utility.Context;
            Accelerator accelerator = ConvolutionalNeuralNetwork.Utility.Accelerator;

            var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<ScalingLayerInfo>>(ForwardKernal);

            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = accelerator.Allocate1D(new ScalingLayerInfo[] { Infos(i) });
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceInputs[i, j] = _inputs[i, j].Allocate(accelerator);
                }
            }

            for (int i = 0; i < _inputDimensions; i++)
            {
                Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
                for (int j = 0; j < _batchSize; j++)
                {
                    _deviceOutputs[i, j] = _outputs[i, j].AllocateEmpty(accelerator);

                    forwardKernal(index, _deviceInputs[i, j].View, _deviceOutputs[i, j].View, _deviceInfos[i].View);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _outputs[i, j].CopyFromBuffer(_deviceOutputs[i, j]);
                    _deviceOutputs[i, j].Dispose();
                    _deviceInputs[i, j].Dispose();
                }
                _deviceInfos[i].Dispose();
            }
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <summary>
        /// Sets all outgoing <see cref="FeatureMap"/>'s to have the same static dimensions.
        /// </summary>
        /// <param name="width">The width of the outgoing <see cref="FeatureMap"/>s.</param>
        /// <param name="length">The length of the outgoing <see cref="FeatureMap"/>s.</param>
        public void SetDimensions(int width, int length)
        {
            _outputWidth = width;
            _outputLength = length;
        }

        /// <summary>
        /// Sets the scales to rescale the incoming <see cref="FeatureMap"/>s.
        /// </summary>
        /// <param name="width">The ratio of the outgoing <see cref="FeatureMap"/>'s width to the incoming <see cref="FeatureMap"/>'s width.</param>
        /// <param name="length">The ratio of the outgoing <see cref="FeatureMap"/>'s length to the incoming <see cref="FeatureMap"/>'s length.</param>
        public void SetScale(float width, float length)
        {
            _scaleWidth = width;
            _scaleLength = length;
        }

        /// <inheritdoc/>
        public override (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] inputs, FeatureMap[,] outGradients)
        {
            _outputDimensions = _inputDimensions = inputs.GetLength(0);
            _inputs = inputs;
            _batchSize = inputs.GetLength(1);
            _layerInfos = new ILayerInfo[_inputDimensions];
            _outGradients = outGradients;
            base._outputs = new FeatureMap[_outputDimensions, _batchSize];
            _inGradients = new FeatureMap[_outputDimensions, _batchSize];

            _deviceInputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];
            _deviceInGradients = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions, _batchSize];
            _deviceOutputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions, _batchSize];
            _deviceOutGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions, _batchSize];
            _deviceInfos = new MemoryBuffer1D<ScalingLayerInfo, Stride1D.Dense>[_inputDimensions];

            for (int i = 0; i < _inputDimensions; i++)
            {
                int outputWidth, outputLength;
                float scaleWidth, scaleLength;
                if (_outputWidth == 0)
                {
                    if (_scaleWidth == 0)
                    {
                        //If the scaling is not set, Scaling defaults to making all FeatureMaps the same size.
                        outputWidth = inputs[0, 0].Width;
                        outputLength = inputs[0, 0].Length;
                        scaleWidth = outputWidth / (float)inputs[i, 0].Width;
                        scaleLength = outputLength / (float)inputs[i, 0].Length;
                    }
                    else
                    {
                        outputWidth = (int)(inputs[i, 0].Width * _scaleWidth);
                        outputLength = (int)(inputs[i, 0].Length * _scaleLength);
                        scaleWidth = _scaleWidth;
                        scaleLength = _scaleLength;
                    }
                }
                else
                {
                    outputWidth = _outputWidth;
                    outputLength = _outputLength;
                    scaleWidth = _outputWidth / (float)inputs[i, 0].Width;
                    scaleLength = _outputLength / (float)inputs[i, 0].Length;
                }

                ILayerInfo layer = _layerInfos[i] = new ScalingLayerInfo()
                {
                    InputWidth = inputs[i, 0].Width,
                    InverseKSquared = scaleWidth * scaleLength,
                    InputLength = inputs[i, 0].Length,
                    OutputWidth = outputWidth,
                    OutputLength = outputLength,
                    InvWidthScaling = 1 / scaleWidth,
                    InvLengthScaling = 1 / scaleLength
                };

                for (int j = 0; j < _batchSize; j++)
                {
                    _outGradients[i, j] = new FeatureMap(layer.InputWidth, layer.InputLength);
                    base._outputs[i, j] = new FeatureMap(outputWidth, outputLength);
                }
            }
            return (_outputs, _inGradients);
        }

        private static void BackwardsKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<float> outGradient, ArrayView<ScalingLayerInfo> info)
        {
            int inGradientIndex = info[0].OutputIndex(index.X, index.Y);

            float minX = index.X * info[0].InvWidthScaling;
            float maxX = minX + info[0].InvWidthScaling;
            float minY = index.Y * info[0].InvLengthScaling;
            float maxY = minY + info[0].InvLengthScaling;
            for (int j = (int)minY; j < (int)MathF.Ceiling(maxY); j++)
            {
                for (int i = (int)minX; i < (int)MathF.Ceiling(maxX); i++)
                {
                    float width = MathF.Min(MathF.Min(1, MathF.Min(i + 1 - minX, maxX - i)), maxX - minX);
                    float height = MathF.Min(MathF.Min(1, MathF.Min(j + 1 - minY, maxY - j)), maxY - minY);
                    Atomic.Add(ref outGradient[3 * info[0].InputIndex(i, j) + index.Z], width * height * inGradient[inGradientIndex][index.Z] * info[0].InverseKSquared);
                }
            }
        }

        private static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> output, ArrayView<ScalingLayerInfo> info)
        {
            Color color = new(0);
            int outputIndex = info[0].OutputIndex(index.X, index.Y);

            float minX = index.X * info[0].InvWidthScaling;
            float maxX = minX + info[0].InvWidthScaling;
            float minY = index.Y * info[0].InvLengthScaling;
            float maxY = minY + info[0].InvLengthScaling;
            for (int j = (int)minY; j < (int)MathF.Ceiling(maxY); j++)
            {
                for (int i = (int)minX; i < (int)MathF.Ceiling(maxX); i++)
                {
                    float width = MathF.Min(MathF.Min(1, MathF.Min(i + 1 - minX, maxX - i)), maxX - minX);
                    float height = MathF.Min(MathF.Min(1, MathF.Min(j + 1 - minY, maxY - j)), maxY - minY);
                    color += width * height * input[info[0].InputIndex(i, j)];
                }
            }

            color *= info[0].InverseKSquared;
            output[outputIndex] = color;
        }

        /// <summary>
        /// Gets the <see cref="ScalingLayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="ScalingLayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="ScalingLayerInfo"/> corresponding to an input dimension.</returns>
        private ScalingLayerInfo Infos(int index)
        {
            return (ScalingLayerInfo)_layerInfos[index];
        }

        private readonly struct ScalingLayerInfo : ILayerInfo
        {
            public int FilterSize => throw new NotImplementedException();
            public int InputLength { get; init; }
            public int InputWidth { get; init; }
            public float InverseKSquared { get; init; }
            public float InvLengthScaling { get; init; }
            public float InvWidthScaling { get; init; }
            public int OutputLength { get; init; }
            public int OutputWidth { get; init; }
            public int Stride => throw new NotImplementedException();
            public int InputIndex(int x, int y)
            {
                return y * InputWidth + x;
            }

            public int OutputIndex(int x, int y)
            {
                return y * OutputWidth + x;
            }
        }
    }
}