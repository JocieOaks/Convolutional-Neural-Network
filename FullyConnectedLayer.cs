using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

public class FullyConnectedLayer : Layer
{
    private FeatureMap _matrix;

    public FullyConnectedLayer(ref FeatureMap[,] input, int outputDimensions) : base(1, 1, ref input, -input.GetLength(0) / outputDimensions)
    {
        float variance = 0.666f / (_inputDimensions + _outputDimensions);
        float stdDev = MathF.Sqrt(variance);
        _matrix = new FeatureMap(_inputDimensions, _outputDimensions);
        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _outputDimensions; j++)
            {
                _matrix[i, j] = Color.RandomGauss(0, stdDev);
            }
        }
    }

    public override string Name => "Fully Connected Layer";

    public override FeatureMap[,] Backwards(FeatureMap[,] inputs, FeatureMap[,] inGradients, float learningRate)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using Accelerator accelerator = context.CreateCudaAccelerator(0);
            MemoryBuffer1D<float, Stride1D.Dense>[,] deviceOutGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions, _batchSize];
            MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceInputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];
            MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceInGradients = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions, _batchSize];
            for (int j = 0; j < _batchSize; j++)
            {
                for (int i = 0; i < _inputDimensions; i++)
                {

                    deviceOutGradients[i, j] = _outGradients[i, j].AllocateFloat(accelerator);
                    deviceInputs[i, j] = inputs[i, j].Allocate(accelerator);
                }
                for (int i = 0; i < _outputDimensions; i++)
                {
                    deviceInGradients[i, j] = inGradients[i, j].Allocate(accelerator);
                }
            }

            MemoryBuffer1D<float, Stride1D.Dense>[,] deviceMultiplierGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions, _outputDimensions];

            for (int i = 0; i < _inputDimensions; i++)
            {
                using MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense> deviceLayerInfo = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });
                Index3D index = new Index3D(_layerInfos[i].InputWidth, _layerInfos[i].InputLength, 3);
                for (int j = 0; j < _outputDimensions; j++)
                {
                    Color multiplier = _matrix[i, j];
                    using MemoryBuffer1D<float, Stride1D.Dense> deviceMultiplier = accelerator.Allocate1D(multiplier.ToArray());
                    deviceMultiplierGradients[i,j] = accelerator.Allocate1D<float>(3);
                    for (int k = 0; k < _batchSize; k++)
                    {
                        using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = inputs[i, k].Allocate(accelerator);
                        Action<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<float>, ArrayView<SingleLayerInfo>> backwardsOutKernal =
                            accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<float>, ArrayView<SingleLayerInfo>>(BackwardsOutKernal);

                        Action<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>> backwardsGradientKernal =
                            accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(BackwardsGradientKernal);

                        backwardsOutKernal(index, deviceInput.View, deviceOutGradients[j, k].View, deviceMultiplier.View, deviceLayerInfo.View);
                    }
                }
            }

            accelerator.Synchronize();


            for (int j = 0; j < _batchSize; j++)
            {
                for (int i = 0; i < _inputDimensions; i++)
                {
                    _outGradients[i, j].CopyFromBuffer(deviceOutGradients[i, j]);
                    deviceOutGradients[i, j].Dispose();
                    deviceInputs[i, j].Dispose();
                }
                for (int i = 0; i < _outputDimensions; i++)
                {
                    deviceInGradients[i, j].Dispose();
                }
            }

            for(int i = 0; i < _inputDimensions; i++)
            {
                for(int j = 0; j < _outputDimensions; j++)
                {
                    _matrix[i, j] += (Color)deviceMultiplierGradients[i, j];
                }
            }
        }
        return _outGradients;
    }

    public override FeatureMap[,] Forward(FeatureMap[,] inputs)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using Accelerator accelerator = context.CreateCudaAccelerator(0);
            MemoryBuffer1D<float, Stride1D.Dense>[,] deviceOutputs = new MemoryBuffer1D<float, Stride1D.Dense>[_outputDimensions, _batchSize];
            MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceInputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];

            for (int j = 0; j < _batchSize; j++)
            {
                for (int i = 0; i < _outputDimensions; i++)
                {

                    deviceOutputs[i, j] = _outputs[i, j].AllocateFloat(accelerator);
                }


                for (int i = 0; i < _inputDimensions; i++)
                {

                    deviceInputs[i, j] = inputs[i, j].Allocate(accelerator);
                }
            }

            for (int i = 0; i < _inputDimensions; i++)
            {
                using MemoryBuffer1D<SingleLayerInfo, Stride1D.Dense> deviceLayerInfo = accelerator.Allocate1D(new SingleLayerInfo[] { Infos(i) });
                Index3D index = new Index3D(_layerInfos[i].InputWidth, _layerInfos[i].InputLength, 3);
                for (int j = 0; j < _outputDimensions; j++)
                {
                    Color multiplier = _matrix[i, j];
                    using MemoryBuffer1D<float, Stride1D.Dense> deviceMultiplier = accelerator.Allocate1D(multiplier.ToArray());

                    for (int k = 0; k < _batchSize; k++)
                    {
                        Action<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<float>, ArrayView<SingleLayerInfo>> forwardKernal =
                            accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<float>, ArrayView<SingleLayerInfo>>(ForwardKernal);

                        forwardKernal(index, deviceInputs[i,k].View, deviceOutputs[j, k].View, deviceMultiplier.View, deviceLayerInfo.View);
                    }
                }
            }

            accelerator.Synchronize();

            for (int j = 0; j < _batchSize; j++)
            {
                for (int i = 0; i < _outputDimensions; i++)
                {

                    _outputs[i, j].CopyFromBuffer(deviceOutputs[i, j]);
                    deviceOutputs[i, j].Dispose();
                }

                for(int i = 0; i < _inputDimensions; i++)
                {
                    deviceInputs[i, j].Dispose();
                }
            }
        }
        return _outputs;
    }

    private static void BackwardsOutKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<float> multiplier, ArrayView<float> outGradient, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        Atomic.Add(ref outGradient[mapsIndex * 3 + index.Z], inGradient[mapsIndex][index.Z] * multiplier[index.Z]);
    }

    private static void BackwardsGradientKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<Color> input, ArrayView<float> multiplierGradient, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        Atomic.Add(ref multiplierGradient[index.Z], inGradient[mapsIndex][index.Z] * input[mapsIndex][index.Z]);
    }

    private SingleLayerInfo Infos(int index)
    {
        return (SingleLayerInfo)_layerInfos[index];
    }

    private static void ForwardKernal(Index3D index, ArrayView<Color> input, ArrayView<float> output, ArrayView<float> multiplier, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        Atomic.Add(ref output[mapsIndex * 3 + index.Z], input[mapsIndex][index.Z] * multiplier[index.Z]);
    }
}