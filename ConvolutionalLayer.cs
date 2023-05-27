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

    protected FeatureMap[][] Convoluted => _outputs;

    protected float[][] _kernalGradient;

    [JsonProperty]
    protected Color[][] _kernals;

    public ConvolutionalLayer(int kernalSize, int stride, ref FeatureMap[][] input) : base(kernalSize, stride, ref input)
    {
        //Setup kernals and kernal gradients
        _kernals = new Color[_dimensions][];
        _kernalGradient = new float[_dimensions][];

        float variance = 0.333f / (_dimensions * kernalSize * kernalSize);
        float stdDev = MathF.Sqrt(variance);


        for (int i = 0; i < _dimensions; i++)
        {
            _kernals[i] = new Color[kernalSize * kernalSize];
            _kernalGradient[i] = new float[kernalSize * kernalSize * 3];
            for (int j = 0; j < kernalSize * kernalSize; j++)
            {
                _kernals[i][j] = Color.RandomGauss(0, stdDev);
            }
        }
    }

    public override FeatureMap[][] Backwards(FeatureMap[][] inputs, FeatureMap[][] inGradients, float learningRate)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using Accelerator accelerator = context.CreateCudaAccelerator(0);
            
            MemoryBuffer1D<float, Stride1D.Dense>[,] deviceOutGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_dimensions, inputs[0].Length];
            MemoryBuffer1D<float, Stride1D.Dense>[] deviceKernalGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_dimensions];

            for (int i = 0; i < _dimensions; i++)
            {
                deviceKernalGradients[i] = accelerator.Allocate1D<float>(_kernalGradient[i].Length);
                for (int j = 0; j < inputs[i].Length; j++)
                {
                    using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = inputs[i][j].Allocate(accelerator);
                    deviceOutGradients[i, j] = InitializeBackwardsKernal(i, deviceInput, inGradients[i][j], accelerator, deviceKernalGradients[i]);
                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _dimensions; i++)
            {
                for (int j = 0; j < inputs[i].Length; j++)
                {
                    _outGradients[i][j].CopyFromBuffer(deviceOutGradients[i, j]);
                    deviceOutGradients[i, j].Dispose();
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

    protected MemoryBuffer1D<float, Stride1D.Dense> InitializeBackwardsKernal(int dimension, MemoryBuffer1D<Color, Stride1D.Dense> deviceInput, FeatureMap inGradient, Accelerator accelerator, MemoryBuffer1D<float, Stride1D.Dense> deviceKernalGradient)
    {
        MemoryBuffer1D<float, Stride1D.Dense> deviceOutGradient = accelerator.Allocate1D<float>(_layerInfos[dimension].InputArea * 3);
        using MemoryBuffer1D<Color, Stride1D.Dense> deviceKernal = accelerator.Allocate1D(_kernals[dimension]);
        using MemoryBuffer1D<Color, Stride1D.Dense> deviceInGradient = inGradient.Allocate(accelerator);
        using MemoryBuffer1D<LayerInfo, Stride1D.Dense> deviceLayerInfo = accelerator.Allocate1D(new LayerInfo[] { _layerInfos[dimension] });

        Action<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>> backWardsKernal =
            accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<float>, ArrayView<LayerInfo>>(BackwardsKernal);

        Index3D index = new(inGradient.Width, inGradient.Length, 3);

        backWardsKernal(index, deviceInput.View, deviceKernal.View, deviceInGradient.View, deviceOutGradient.View, deviceKernalGradient.View, deviceLayerInfo.View);
        
        return deviceOutGradient;
    }

    public override FeatureMap[][] Forward(FeatureMap[][] inputs)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using Accelerator accelerator = context.CreateCudaAccelerator(0);
            
            MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceConvoluted = new MemoryBuffer1D<Color, Stride1D.Dense>[_dimensions, inputs[0].Length];

            for (int i = 0; i < _dimensions; i++)
            {
                for (int j = 0; j < inputs[i].Length; j++)
                {
                    using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = inputs[i][j].Allocate(accelerator);
                    deviceConvoluted[i, j] = InitializeForwardKernal(i, deviceInput, accelerator);

                }
            }

            accelerator.Synchronize();

            for (int i = 0; i < _dimensions; i++)
            {
                for (int j = 0; j < inputs[i].Length; j++)
                {
                    Convoluted[i][j].CopyFromBuffer(deviceConvoluted[i, j]);
                    deviceConvoluted[i, j].Dispose();
                }
            }
        }

        return Convoluted;
    }

    protected MemoryBuffer1D<Color, Stride1D.Dense> InitializeForwardKernal(int dimension, MemoryBuffer1D<Color, Stride1D.Dense> deviceInput, Accelerator accelerator)
    {
        MemoryBuffer1D<Color, Stride1D.Dense> deviceConvoluted = accelerator.Allocate1D<Color>(_layerInfos[dimension].OutputArea);
        using MemoryBuffer1D<Color, Stride1D.Dense> deviceKernal = accelerator.Allocate1D(_kernals[dimension]);
        using MemoryBuffer1D<LayerInfo, Stride1D.Dense> deviceLayerInfo = accelerator.Allocate1D(new LayerInfo[] { _layerInfos[dimension] });

        Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<LayerInfo>> forwardKernal =
            accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<LayerInfo>>(ForwardKernal);

        Index2D index = new(_layerInfos[dimension].OutputWidth, _layerInfos[dimension].OutputLength);

        forwardKernal(index, deviceInput.View, deviceConvoluted.View, deviceKernal.View, deviceLayerInfo.View);
        return deviceConvoluted;
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

    private static int FloatIndex(int index, int rgb)
    {
        return index * 3 + rgb;
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
}