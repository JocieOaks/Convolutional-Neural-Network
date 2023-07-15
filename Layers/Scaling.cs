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

        private static readonly Action<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<ScalingLayerInfo>> s_backwardsAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<float>, ArrayView<float>, ArrayView<ScalingLayerInfo>>(BackwardsKernel);

        /// <inheritdoc/>
        public override void Backwards(float learningRatee, float firstMomentDecay, float secondMomentDecay)
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index3D index = new(Infos(i).OutputWidth, Infos(i).OutputLength, 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    _buffers.OutGradientsColor[i, j].SubView(0, Infos(i).InputArea).MemSetToZero();
                    s_backwardsAction(index, _buffers.InGradientsFloat[i, j], _buffers.OutGradientsFloat[i, j], _deviceInfos[i].View);
                }
            }
            Synchronize();
        }

        private readonly static Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<ScalingLayerInfo>> s_forwardAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<ScalingLayerInfo>>(ForwardKernel);

        /// <inheritdoc/>
        public override void Forward()
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
                for (int j = 0; j < _batchSize; j++)
                {
                    s_forwardAction(index, _buffers.InputsColor[i, j], _buffers.OutputsColor[i, j], _deviceInfos[i].View);
                }
            }
            Synchronize();
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
        public override FeatureMap[,] Startup(FeatureMap[,] inputs, IOBuffers buffers)
        {
            _outputDimensions = _inputDimensions = inputs.GetLength(0);
            _buffers = buffers;
            _batchSize = (uint)inputs.GetLength(1);
            _layerInfos = new ILayerInfo[_inputDimensions];
            _outputs = new FeatureMap[_outputDimensions, _batchSize];

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

                _layerInfos[i] = new ScalingLayerInfo()
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
                    _outputs[i, j] = new FeatureMap(outputWidth, outputLength);
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
                buffers.OutputDimensionArea(i, _outputs[i, 0].Area);

            _deviceInfos = new MemoryBuffer1D<ScalingLayerInfo, Stride1D.Dense>[_inputDimensions];
            for (int i = 0; i < _inputDimensions; i++)
            {
                _deviceInfos[i] = GPU.GPUManager.Accelerator.Allocate1D(new ScalingLayerInfo[] { Infos(i) });
            }

            return _outputs;
        }

        /// <summary>
        /// An ILGPU kernel to calculate the gradients for backpropagating the previous layer.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
        /// <param name="inGradient">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the incoming
        /// gradient from the following <see cref="Layer"/>.</param>
        /// <param name="outGradient">An <see cref="ArrayView1D{T, TStride}"/> of floats to sum the outgoing gradient.
        /// Because <see cref="Color"/> cannot be summed atomically, every three floats represents a single
        /// <see cref="Color"/> in the gradient.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void BackwardsKernel(Index3D index, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<ScalingLayerInfo> info)
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
                    Atomic.Add(ref outGradient[3 * info[0].InputIndex(i, j) + index.Z], width * height * inGradient[3 * inGradientIndex + index.Z] * info[0].InverseKSquared);
                }
            }
        }

        /// <summary>
        /// An <see cref="ILGPU"/> kernel for resampling a <see cref="FeatureMap"/>.
        /// </summary>
        /// <param name="index">The index of the current kernel calculation to be made.</param>
        /// <param name="input">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s containing the input from the
        /// previous <see cref="Layer"/>.</param>
        /// <param name="output">An <see cref="ArrayView1D{T, TStride}"/> of <see cref="Color"/>s to set for the outgoing
        /// resampled <see cref="FeatureMap"/>.</param>
        /// <param name="info">The <see cref="LayerInfo"/> for the current dimension at the first index of an <see cref="ArrayView1D{T, TStride}"/>.</param>
        private static void ForwardKernel(Index2D index, ArrayView<Color> input, ArrayView<Color> output, ArrayView<ScalingLayerInfo> info)
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

        /// <summary>
        /// The <see cref="ScalingLayerInfo"/> struct contains a variety of data about <see cref="Scaling"/> layers
        /// and <see cref="FeatureMap"/>s for use by an <see cref="ILGPU"/> kernel.
        /// </summary>
        public readonly struct ScalingLayerInfo : ILayerInfo
        {
            /// <inheritdoc/>
            public int FilterSize => throw new NotImplementedException(); //Should not be used.

            /// <inheritdoc/>
            public int InputLength { get; init; }

            /// <inheritdoc/>
            public int InputWidth { get; init; }

            public int InputArea => InputLength * InputWidth;

            /// <inheritdoc/>
            public float InverseKSquared { get; init; }

            /// <value>The ratio of the <see cref="OutputLength"/> to the <see cref="InputLength"/>.</value>
            public float InvLengthScaling { get; init; }

            /// <value>The ratio of the <see cref="OutputWidth"/> to the <see cref="InputWidth"/>.</value>
            public float InvWidthScaling { get; init; }

            /// <inheritdoc/>
            public int OutputLength { get; init; }

            /// <inheritdoc/>
            public int OutputWidth { get; init; }

            public int OutputArea => OutputLength * OutputWidth;

            /// <inheritdoc/>
            public int Stride => throw new NotImplementedException();   //Should not be used.

            /// <summary>
            /// Calculates the single dimensional array index for a flattened input <see cref="FeatureMap"/>.
            /// </summary>
            /// <param name="x">The x coordinate of the desired index.</param>
            /// <param name="y">The y coordinate of the desired index.</param>
            /// <returns>Returns the index corresponding to (<paramref name="x"/>, <paramref name="y"/>).</returns>
            public int InputIndex(int x, int y)
            {
                return y * InputWidth + x;
            }

            /// <summary>
            /// Calculates the single dimensional array index for a flattened output <see cref="FeatureMap"/>.
            /// </summary>
            /// <param name="x">The x coordinate of the desired index.</param>
            /// <param name="y">The y coordinate of the desired index.</param>
            /// <returns>Returns the index corresponding to (<paramref name="x"/>, <paramref name="y"/>).</returns>
            public int OutputIndex(int x, int y)
            {
                return y * OutputWidth + x;
            }
        }
    }
}