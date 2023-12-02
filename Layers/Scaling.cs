using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="Scaling"/> class is a <see cref="Layer"/> for upsampling or downsampling a <see cref="FeatureMap"/>.
    /// </summary>
    public class Scaling : Layer
    {
        /*private MemoryBuffer1D<ScalingLayerInfo, Stride1D.Dense>[] _deviceInfos;
        [JsonProperty] private int _outputLength;
        [JsonProperty] private int _outputWidth;
        private float _scaleLength;
        private float _scaleWidth;

        /// <inheritdoc/>
        public override string Name => "Scaling Layer";

        private static readonly Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<ScalingLayerInfo>> s_backwardsAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<ScalingLayerInfo>>(BackwardsKernel);

        /// <inheritdoc/>
        public override void Backwards(float learningRatee, float firstMomentDecay, float secondMomentDecay)
        {
            for (int i = 0; i < _inputShape.Dimensions; i++)
            {
                Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
                for (int j = 0; j < batchSize; j++)
                {
                    _buffers.OutGradient[i, j].SubView(0, Infos(i).InputArea).MemSetToZero();
                    s_backwardsAction(index, _buffers.InGradient[i, j], _buffers.OutGradient[i, j], _deviceInfos[i].View);
                }
            }
            Synchronize();
        }

        private readonly static Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<ScalingLayerInfo>> s_forwardAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<ScalingLayerInfo>>(ForwardKernel);

        /// <inheritdoc/>
        public override void Forward()
        {
            for (int i = 0; i < _inputShape.Dimensions; i++)
            {
                Index2D index = new(Infos(i).OutputWidth, Infos(i).OutputLength);
                for (int j = 0; j < batchSize; j++)
                {
                    s_forwardAction(index, _buffers.Input[i, j], _buffers.Output[i, j], _deviceInfos[i].View);
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
        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, uint batchSize)
        {

        if (_ready)
                return _outputShape;
            _ready = true;
            _outputShape.Dimensions = _inputShape.Dimensions = inputShapes.Length;
            _buffers = buffers;
            batchSize = batchSize;
            _layerInfos = new ILayerInfo[_inputShape.Dimensions];
            _outputShapes = new Shape[_outputShape.Dimensions];

            for (int i = 0; i < _inputShape.Dimensions; i++)
            {
                int outputWidth, outputLength;
                float scaleWidth, scaleLength;
                if (_outputWidth == 0)
                {
                    if (_scaleWidth == 0)
                    {
                        //If the scaling is not set, Scaling defaults to making all FeatureMaps the same size.
                        outputWidth = inputShapes[0].Width;
                        outputLength = inputShapes[0].Length;
                        scaleWidth = outputWidth / (float)inputShapes[i].Width;
                        scaleLength = outputLength / (float)inputShapes[i].Length;
                    }
                    else
                    {
                        outputWidth = (int)(inputShapes[i].Width * _scaleWidth);
                        outputLength = (int)(inputShapes[i].Length * _scaleLength);
                        scaleWidth = _scaleWidth;
                        scaleLength = _scaleLength;
                    }
                }
                else
                {
                    outputWidth = _outputWidth;
                    outputLength = _outputLength;
                    scaleWidth = _outputWidth / (float)inputShapes[i].Width;
                    scaleLength = _outputLength / (float)inputShapes[i].Length;
                }

                _layerInfos[i] = new ScalingLayerInfo()
                {
                    InputWidth = inputShapes[i].Width,
                    InverseKSquared = scaleWidth * scaleLength,
                    InputLength = inputShapes[i].Length,
                    OutputWidth = outputWidth,
                    OutputLength = outputLength,
                    InvWidthScaling = 1 / scaleWidth,
                    InvLengthScaling = 1 / scaleLength
                };

                inputShapes[i] = new Shape(outputWidth, outputLength);
            }

            for (int i = 0; i < _outputShape.Dimensions; i++)
                buffers.OutputDimensionArea(i, _outputShapes[i].Area);

            _deviceInfos = new MemoryBuffer1D<ScalingLayerInfo, Stride1D.Dense>[_inputShape.Dimensions];
            for (int i = 0; i < _inputShape.Dimensions; i++)
            {
                _deviceInfos[i] = GPU.GPUManager.Accelerator.Allocate1D(new ScalingLayerInfo[] { Infos(i) });
            }

            return _outputShapes;
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
        private static void BackwardsKernel(Index2D index, ArrayView<float> inGradient, ArrayView<float> outGradient, ArrayView<ScalingLayerInfo> info)
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
                    Atomic.Add(ref outGradient[info[0].InputIndex(i, j)], width * height * inGradient[inGradientIndex] * info[0].InverseKSquared);
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
        private static void ForwardKernel(Index2D index, ArrayView<float> input, ArrayView<float> output, ArrayView<ScalingLayerInfo> info)
        {
            float color = 0;
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
        }*/
        public override string Name => throw new NotImplementedException();

        public override void Backwards(int batchSize, bool update)
        {
            throw new NotImplementedException();
        }

        public override void Forward(int batchSize)
        {
            throw new NotImplementedException();
        }

        public override Shape Startup(Shape inputShapes, IOBuffers buffers, int maxBatchSize)
        {
            throw new NotImplementedException();
        }
    }
}