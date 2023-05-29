using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System.Linq.Expressions;

public class FullyConnectedLayer : Layer
{
    private FeatureMap _redMatrix;
    private FeatureMap _greenMatrix;
    private FeatureMap _blueMatrix;

    public FullyConnectedLayer(ref FeatureMap[,] input, int outputDimensions) : base(1, 1, ref input, -input.GetLength(0) / outputDimensions)
    {
        float variance = 0.666f / (_inputDimensions + _outputDimensions);
        float stdDev = MathF.Sqrt(variance);
        _redMatrix = new FeatureMap(_inputDimensions, _outputDimensions);
        _greenMatrix = new FeatureMap(_inputDimensions, _outputDimensions);
        _blueMatrix = new FeatureMap(_inputDimensions, _outputDimensions);
        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _outputDimensions; j++)
            {
                _redMatrix[i, j] = Color.RandomGauss(0, stdDev);
                _greenMatrix[i, j] = Color.RandomGauss(0, stdDev);
                _blueMatrix[i, j] = Color.RandomGauss(0, stdDev);
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
                    using MemoryBuffer1D<Color, Stride1D.Dense> deviceMultiplier = accelerator.Allocate1D(new Color[] { _redMatrix[i, j], _greenMatrix[i,j], _blueMatrix[i, j] });
                    deviceMultiplierGradients[i,j] = accelerator.Allocate1D<float>(9);
                    for (int k = 0; k < _batchSize; k++)
                    {
                        Action<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>> backwardsOutKernal =
                            accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(BackwardsOutKernal);

                        Action<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>> backwardsGradientKernal =
                            accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<SingleLayerInfo>>(BackwardsGradientKernal);

                        backwardsOutKernal(index, deviceInGradients[j, k].View, deviceMultiplier.View, deviceOutGradients[i, k].View, deviceLayerInfo.View);
                        backwardsGradientKernal(index, deviceInGradients[j, k].View, deviceInputs[i, k].View, deviceMultiplierGradients[i, j].View, deviceLayerInfo.View);
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

            float[] multiplierGradients = new float[9];

            for(int i = 0; i < _inputDimensions; i++)
            {
                for(int j = 0; j < _outputDimensions; j++)
                {
                    deviceMultiplierGradients[i, j].CopyToCPU(multiplierGradients);
                    _redMatrix[i, j] -= new Color(multiplierGradients[0], multiplierGradients[1], multiplierGradients[2]) * learningRate;
                    _greenMatrix[i, j] -= new Color(multiplierGradients[3], multiplierGradients[4], multiplierGradients[5]) * learningRate;
                    _blueMatrix[i, j] -= new Color(multiplierGradients[6], multiplierGradients[7], multiplierGradients[8]) * learningRate;
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
                    using MemoryBuffer1D<Color, Stride1D.Dense> deviceMultiplier = accelerator.Allocate1D(new Color[] { _redMatrix[i, j], _greenMatrix[i, j], _blueMatrix[i, j] });

                    for (int k = 0; k < _batchSize; k++)
                    {
                        Action<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<Color>, ArrayView<SingleLayerInfo>> forwardKernal =
                            accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<float>, ArrayView<Color>, ArrayView<SingleLayerInfo>>(ForwardKernal);

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

    private static void BackwardsOutKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<Color> multiplier, ArrayView<float> outGradient, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        float transposeDot = 0;
        for(int i = 0; i < 3; i++)
        {
            transposeDot += inGradient[mapsIndex][i] * multiplier[i][index.Z];
        }
        Atomic.Add(ref outGradient[mapsIndex * 3 + index.Z], transposeDot);
    }

    private static void BackwardsGradientKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<Color> input, ArrayView<float> multiplierGradient, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        for (int i = 0; i < 3; i++)
        {
            Atomic.Add(ref multiplierGradient[index.Z * 3 + i], inGradient[mapsIndex][index.Z] * input[mapsIndex][i]);
        }
    }

    private SingleLayerInfo Infos(int index)
    {
        return (SingleLayerInfo)_layerInfos[index];
    }

    private static void ForwardKernal(Index3D index, ArrayView<Color> input, ArrayView<float> output, ArrayView<Color> multiplier, ArrayView<SingleLayerInfo> info)
    {
        int mapsIndex = info[0].Index(index.X, index.Y);
        Atomic.Add(ref output[mapsIndex * 3 + index.Z], Color.Dot(input[mapsIndex], multiplier[index.Z]));
    }
}