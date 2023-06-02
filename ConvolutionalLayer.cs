// See https://aka.ms/new-console-template for more information
#nullable disable

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using Newtonsoft.Json;
using System;
using System.Runtime.Serialization;
using static System.Net.Mime.MediaTypeNames;

[Serializable]
public class ConvolutionalLayer : Layer
{
    protected const int CLAMP = 1;

    protected const float LEARNINGMULTIPLIER = 1f;

    protected float[][] _kernalGradient;
    [JsonProperty]
    protected Color[][] _kernals;

    public override string Name => "Convolutional Layer";

    public ConvolutionalLayer(int kernalSize, int stride, ref FeatureMap[,] input, int outputDimensionsMultiplier) : base(kernalSize, stride, ref input, outputDimensionsMultiplier)
    {
        if(outputDimensionsMultiplier < 1)
        {
            throw new ArgumentException("Output Dimension Multiplier must be greater than 0. To decrease dimensions, use Fully Connected Layer");
        }

        //Setup kernals and kernal gradients
        _kernals = new Color[_outputDimensions][];
        _kernalGradient = new float[_outputDimensions][];

        float variance = 0.6666f / (_outputDimensions * kernalSize * kernalSize + _inputDimensions * kernalSize * kernalSize);
        float stdDev = MathF.Sqrt(variance);


        for (int i = 0; i < _outputDimensions; i++)
        {
            _kernals[i] = new Color[kernalSize * kernalSize];
            _kernalGradient[i] = new float[kernalSize * kernalSize * 3];
            for (int j = 0; j < kernalSize * kernalSize; j++)
            {
                _kernals[i][j] = Color.RandomGauss(0, stdDev);
            }
        }
    }

    protected FeatureMap[,] Convoluted => _outputs;
    public override FeatureMap[,] Backwards(FeatureMap[,] inputs, FeatureMap[,] inGradients, float learningRate)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using Accelerator accelerator = context.CreateCudaAccelerator(0);
            
            MemoryBuffer1D<float, Stride1D.Dense>[,] deviceOutGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions, _batchSize];
            MemoryBuffer1D<float, Stride1D.Dense>[] deviceKernalGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_outputDimensions];
            MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceInputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];
            for (int i = 0; i < _inputDimensions; i++)
            {
                for(int j = 0; j < _batchSize; j++)
                {
                    deviceInputs[i, j] = inputs[i, j].Allocate(accelerator);
                    deviceOutGradients[i,j] = accelerator.Allocate1D<float>(Infos(i).InputArea * 3);
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                deviceKernalGradients[i] = accelerator.Allocate1D<float>(_kernalGradient[i].Length);
                for (int j = 0; j < _batchSize; j++)
                {
                    InitializeBackwardsKernal(i, deviceInputs[i % _inputDimensions, j], inGradients[i, j], accelerator, deviceOutGradients[i % _inputDimensions, j], deviceKernalGradients[i]);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _outGradients[i, j].CopyFromBuffer(deviceOutGradients[i, j]);
                    deviceOutGradients[i, j].Dispose();
                    deviceInputs[i, j].Dispose();
                }

                deviceKernalGradients[i].CopyToCPU(_kernalGradient[i]);
                deviceKernalGradients[i].Dispose();

                for (int j = 0; j < _kernalSize * _kernalSize; j++)
                {
                    _kernals[i][j] -= learningRate * LEARNINGMULTIPLIER * new Color(_kernalGradient[i][j * 3], _kernalGradient[i][j * 3 + 1], _kernalGradient[i][j * 3 + 2]).Clamp(CLAMP);
                }
            }
        }

        return _outGradients;
    }

    public void BackwardsKernalOnly(FeatureMap[,] inputs, FeatureMap[,] inGradients, float learningRate)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using Accelerator accelerator = context.CreateCudaAccelerator(0);

            MemoryBuffer1D<float, Stride1D.Dense>[] deviceKernalGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_outputDimensions];
            MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceInputs = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, _batchSize];
            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    deviceInputs[i, j] = inputs[i, j].Allocate(accelerator);
                }
            }

            for (int i = 0; i < _outputDimensions; i++)
            {
                deviceKernalGradients[i] = accelerator.Allocate1D<float>(_kernalGradient[i].Length);
                for (int j = 0; j < _batchSize; j++)
                {
                    InitializeBackwardsKernal(i, deviceInputs[i % _inputDimensions, j], inGradients[i, j], accelerator, deviceKernalGradients[i]);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _inputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    deviceInputs[i, j].Dispose();
                }

                deviceKernalGradients[i].CopyToCPU(_kernalGradient[i]);
                deviceKernalGradients[i].Dispose();

                for (int j = 0; j < _kernalSize * _kernalSize; j++)
                {
                    _kernals[i][j] -= learningRate * LEARNINGMULTIPLIER * new Color(_kernalGradient[i][j * 3], _kernalGradient[i][j * 3 + 1], _kernalGradient[i][j * 3 + 2]).Clamp(CLAMP);
                }
            }
        }
    }

    public override FeatureMap[,] Forward(FeatureMap[,] inputs)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using Accelerator accelerator = context.CreateCudaAccelerator(0);

            MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceConvoluted = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions, _batchSize];

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    using var deviceInput = inputs[i % _inputDimensions, j].Allocate(accelerator);
                    deviceConvoluted[i, j] = InitializeForwardKernal(i, deviceInput, accelerator);

                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    Convoluted[i, j].CopyFromBuffer(deviceConvoluted[i, j]);
                    deviceConvoluted[i, j].Dispose();
                }
            }
        }

        return Convoluted;
    }

    protected static void BackwardsKernal(Index3D index, ArrayView<Color> input, ArrayView<Color> kernal, ArrayView<Color> inGradient, ArrayView<float> outGradient, ArrayView<float> kernalGradient, ArrayView<LayerInfo> info)
    {
        float dL = inGradient[info[0].OutputIndex(index.X, index.Y)][index.Z] * info[0].InverseKSquared;

        for (int j = 0; j < info[0].KernalSize; j++)
        {
            for (int i = 0; i < info[0].KernalSize; i++)
            {
                if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                {
                    int kernalIndex = info[0].KernalIndex(i, j);
                    float dK = dL * input[inputIndex][index.Z];
                    Atomic.Add(ref kernalGradient[FloatIndex(kernalIndex, index.Z)], dK);
                    float dP = dL * kernal[kernalIndex][index.Z];
                    Atomic.Add(ref outGradient[FloatIndex(inputIndex, index.Z)], dP);
                }
            }
        }
    }

    protected static void BackwardsOutKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<Color> kernal, ArrayView<float> outGradient, ArrayView<LayerInfo> info)
    {
        float dL = inGradient[info[0].OutputIndex(index.X, index.Y)][index.Z] * info[0].InverseKSquared;

        for (int j = 0; j < info[0].KernalSize; j++)
        {
            for (int i = 0; i < info[0].KernalSize; i++)
            {
                if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                {
                    int kernalIndex = info[0].KernalIndex(i, j);
                    float dP = dL * kernal[kernalIndex][index.Z];
                    Atomic.Add(ref outGradient[FloatIndex(inputIndex, index.Z)], dP);
                }
            }
        }
    }

    protected static void BackwardsGradientKernal(Index3D index, ArrayView<Color> inGradient, ArrayView<Color> input, ArrayView<float> kernalGradient, ArrayView<LayerInfo> info)
    {
        float dL = inGradient[info[0].OutputIndex(index.X, index.Y)][index.Z] * info[0].InverseKSquared;

        for (int j = 0; j < info[0].KernalSize; j++)
        {
            for (int i = 0; i < info[0].KernalSize; i++)
            {
                if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                {
                    int kernalIndex = info[0].KernalIndex(i, j);
                    float dK = dL * input[inputIndex][index.Z];
                    Atomic.Add(ref kernalGradient[FloatIndex(kernalIndex, index.Z)], dK);
                }
            }
        }
    }

    protected static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> convoluted, ArrayView<Color> kernal, ArrayView<LayerInfo> info)
    {
        Color sum = new();

        for (int j = 0; j < info[0].KernalSize; j++)
        {
            for (int i = 0; i < info[0].KernalSize; i++)
            {
                if (info[0].TryGetInputIndex(index.X, i, index.Y, j, out int inputIndex))
                    sum += kernal[info[0].KernalIndex(i, j)] * input[inputIndex];
            }

        }

        convoluted[info[0].OutputIndex(index.X, index.Y)] = sum * info[0].InverseKSquared;
    }

    private void InitializeBackwardsKernal(int dimension, MemoryBuffer1D<Color, Stride1D.Dense> deviceInput, FeatureMap inGradient, Accelerator accelerator, MemoryBuffer1D<float, Stride1D.Dense> deviceOutGradient, MemoryBuffer1D<float, Stride1D.Dense> deviceKernalGradient)
    {
        using var deviceKernal = accelerator.Allocate1D(_kernals[dimension]);
        using var deviceInGradient = inGradient.Allocate(accelerator);
        using var deviceLayerInfo = accelerator.Allocate1D(new LayerInfo[] { Infos(dimension) });

        var backwardsOutKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsOutKernal);

        var backwardsGradientKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsGradientKernal);

        Index3D index = new(inGradient.Width, inGradient.Length, 3);

        backwardsOutKernal(index, deviceInGradient.View, deviceKernal.View, deviceOutGradient.View, deviceLayerInfo.View);
        backwardsGradientKernal(index, deviceInGradient.View, deviceInput.View, deviceKernalGradient.View, deviceLayerInfo.View);
    }

    private void InitializeBackwardsKernal(int dimension, MemoryBuffer1D<Color, Stride1D.Dense> deviceInput, FeatureMap inGradient, Accelerator accelerator, MemoryBuffer1D<float, Stride1D.Dense> deviceKernalGradient)
    {
        using var deviceInGradient = inGradient.Allocate(accelerator);
        using var deviceLayerInfo = accelerator.Allocate1D(new LayerInfo[] { Infos(dimension) });

        var backwardsGradientKernal = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsGradientKernal);

        Index3D index = new(inGradient.Width, inGradient.Length, 3);

        backwardsGradientKernal(index, deviceInGradient.View, deviceInput.View, deviceKernalGradient.View, deviceLayerInfo.View);
    }

    protected LayerInfo Infos(int index)
    {
        return (LayerInfo)_layerInfos[index % _inputDimensions];
    }

    protected MemoryBuffer1D<Color, Stride1D.Dense> InitializeForwardKernal(int dimension, MemoryBuffer1D<Color, Stride1D.Dense> deviceInput, Accelerator accelerator)
    {
        var deviceConvoluted = accelerator.Allocate1D<Color>(Infos(dimension).OutputArea);
        using var deviceKernal = accelerator.Allocate1D(_kernals[dimension]);
        using var deviceLayerInfo = accelerator.Allocate1D(new LayerInfo[] { Infos(dimension) });

        var forwardKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<LayerInfo>>(ForwardKernal);

        Index2D index = new(Infos(dimension).OutputWidth, Infos(dimension).OutputLength);

        forwardKernal(index, deviceInput.View, deviceConvoluted.View, deviceKernal.View, deviceLayerInfo.View);
        return deviceConvoluted;
    }
    private static int FloatIndex(int index, int rgb)
    {
        return index * 3 + rgb;
    }
}