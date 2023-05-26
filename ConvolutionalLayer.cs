// See https://aka.ms/new-console-template for more information
#nullable disable

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;
using System;
using System.Runtime.Serialization;
using static System.Net.Mime.MediaTypeNames;

[Serializable]
public class ConvolutionalLayer : Layer
{
    protected const int CLAMP = 1;

    protected const float LEARNINGMULTIPLIER = 1f;

    protected FeatureMap[][] _convoluted;

    protected float[][] _dL_dK;
    float[][] _next;

    protected FeatureMap[][] _dL_dPNext;

    protected float _invK2;

    [JsonProperty]
    protected Color[][] _kernals;
    protected int _threadsWorking;

    public ConvolutionalLayer(int kernalSize, int stride, ref FeatureMap[][] input) : base(input.Length, kernalSize, stride)
    {
        _kernals = new Color[_dimensions][];
        _dL_dK = new float[_dimensions][];
        _next = new float[_dimensions][];

        float variance = 0.333f / (_dimensions * kernalSize * kernalSize);
        float stdDev = MathF.Sqrt(variance);

        for (int i = 0; i < _dimensions; i++)
        {
            _kernals[i] = new Color[_kernalSize * _kernalSize];
            _dL_dK[i] = new float[_kernalSize * _kernalSize * 3];
            _next[i] = new float[input[i][0].Area * 3];
            for (int j = 0; j < _kernalSize * _kernalSize; j++)
            {
                _kernals[i][j] = Color.RandomGauss(0, stdDev);
            }
        }

        _convoluted = new FeatureMap[_dimensions][];
        _dL_dPNext = new FeatureMap[_dimensions][];
        for (int i = 0; i < _dimensions; i++)
        {
            Pad(input[i]);
            _convoluted[i] = new FeatureMap[input[i].Length];
            _dL_dPNext[i] = new FeatureMap[input[i].Length];
            for (int j = 0; j < input[i].Length; j++)
            {
                _dL_dPNext[i][j] = new FeatureMap(input[i][j].Width, input[i][j].Length);
                int width = (input[i][j].Width - kernalSize) / stride;
                int length = (input[i][j].Length - kernalSize) / stride;
                _convoluted[i][j] = new FeatureMap(width, length);
            }
        }
        input = _convoluted;

        _invK2 = 1f / (kernalSize * kernalSize);
    }

    public ConvolutionalLayer() : base(0, 0, 0) { }

    public override FeatureMap[][] Backwards(FeatureMap[][] input, FeatureMap[][] dL_dP, float learningRate)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using (Accelerator accelerator = context.CreateCudaAccelerator(0))
            {
                MemoryBuffer1D<float, Stride1D.Dense>[,] deviceDL_dPNext = new MemoryBuffer1D<float, Stride1D.Dense>[_dimensions, input[0].Length];
                MemoryBuffer1D<float, Stride1D.Dense>[] deviceDL_dK = new MemoryBuffer1D<float, Stride1D.Dense>[_dimensions];
                
                for (int i = 0; i < _dimensions; i++)
                {
                    deviceDL_dK[i] = accelerator.Allocate1D<float>(_dL_dK[i].Length);
                    for (int j = 0; j < input[i].Length; j++)
                    {
                        deviceDL_dPNext[i, j] = accelerator.Allocate1D<float>(input[i][j].Area * 3);
                        using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = input[i][j].Allocate(accelerator);
                        using MemoryBuffer1D<Color, Stride1D.Dense> deviceKernal = accelerator.Allocate1D(_kernals[i]);
                        using MemoryBuffer1D<Color, Stride1D.Dense> deviceDL_dP = dL_dP[i][j].Allocate(accelerator);
                        using MemoryBuffer1D<GPUKernalFeatures, Stride1D.Dense> deviceKernalFeatures = 
                            accelerator.Allocate1D(new GPUKernalFeatures[] { new GPUKernalFeatures(input[i][j].Width, dL_dP[i][j].Width, _kernalSize, _stride, _invK2) });

                        Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<float>, ArrayView<GPUKernalFeatures>> backWardsKernal =
                            accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<float>, ArrayView<GPUKernalFeatures>>(BackwardsKernal);

                        Index2D index = new(dL_dP[i][j].Width, dL_dP[i][j].Length);
                            
                        backWardsKernal(index, deviceInput.View, deviceKernal.View, deviceDL_dP.View, deviceDL_dPNext[i, j].View, deviceDL_dK[i].View, deviceKernalFeatures.View);
                        
                    }
                }
                
                accelerator.Synchronize();
                
                for (int i = 0; i < _dimensions; i++)
                {
                    for (int j = 0; j < input[i].Length; j++)
                    {
                        _dL_dPNext[i][j].CopyFromBuffer(deviceDL_dPNext[i,j]);
                        deviceDL_dPNext[i, j].Dispose();
                    }

                    deviceDL_dK[i].CopyToCPU(_dL_dK[i]);
                    deviceDL_dK[i].Dispose();

                    for (int j = 0; j < _kernalSize * _kernalSize; j++)
                    {
                        _kernals[i][j] -= learningRate * LEARNINGMULTIPLIER * new Color(_dL_dK[i][j * 3], _dL_dK[i][j * 3 + 1], _dL_dK[i][j * 3 + 2]).Clamp(CLAMP);
                    }
                }
            }
        }

        return _dL_dPNext;
    }

    public override FeatureMap[][] Forward(FeatureMap[][] input)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using (Accelerator accelerator = context.CreateCudaAccelerator(0))
            {
                MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceConvoluted = new MemoryBuffer1D<Color, Stride1D.Dense>[_dimensions, input[0].Length];
                
                for (int i = 0; i < _dimensions; i++)
                {
                    for (int j = 0; j < input[i].Length; j++)
                    {
                        deviceConvoluted[i, j] = _convoluted[i][j].AllocateEmpty(accelerator);
                        using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = input[i][j].Allocate(accelerator);
                        using MemoryBuffer1D<Color, Stride1D.Dense> deviceKernal = accelerator.Allocate1D(_kernals[i]);
                        using MemoryBuffer1D<GPUKernalFeatures, Stride1D.Dense> deviceKernalFeatures = 
                            accelerator.Allocate1D(new GPUKernalFeatures[] { new GPUKernalFeatures(input[i][j].Width, _convoluted[i][j].Width, _kernalSize, _stride, _invK2) });
                        
                        Action<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<GPUKernalFeatures>> forwardKernal =
                            accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<GPUKernalFeatures>>(ForwardKernal);
                        
                        Index2D index = new(_convoluted[i][j].Width, _convoluted[i][j].Length);
                        
                        forwardKernal(index, deviceInput.View, deviceConvoluted[i, j].View, deviceKernal.View, deviceKernalFeatures.View);

                    }
                }
                
                accelerator.Synchronize();
                
                for (int i = 0; i < _dimensions; i++)
                {
                    for (int j = 0; j < input[i].Length; j++)
                    {
                        _convoluted[i][j].CopyFromBuffer(deviceConvoluted[i, j]);
                        deviceConvoluted[i, j].Dispose();
                    }
                }
            }
        }

        return _convoluted;
    }

    [OnDeserialized]
    public void OnDeserialized(StreamingContext context)
    {
        _invK2 = 1f / (_kernalSize * _kernalSize);
    }
    protected static void BackwardsKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> kernal, ArrayView<Color> dL_dP, ArrayView<float> dL_dPNext, ArrayView<float> dL_dK, ArrayView<GPUKernalFeatures> kF)
    {
        Color dL = dL_dP[index.Y * kF[0].OutputWidth + index.X] * kF[0].InverseKSquared;

        for (int j = 0; j < kF[0].KernalSize; j++)
        {
            int offset = (index.Y * kF[0].Stride + j) * kF[0].InputWidth + index.X * kF[0].Stride;
            for (int i = 0; i < kF[0].KernalSize; i++)
            {
                Color dK = dL * input[offset + i];
                Atomic.Add(ref dL_dK[(j * kF[0].KernalSize + i) * 3], dK.R);
                Atomic.Add(ref dL_dK[(j * kF[0].KernalSize + i) * 3 + 1], dK.G);
                Atomic.Add(ref dL_dK[(j * kF[0].KernalSize + i) * 3 + 2], dK.B);
                Color dP = dL * kernal[j + i * kF[0].KernalSize];
                Atomic.Add(ref dL_dPNext[(offset + i) * 3], dP.R);
                Atomic.Add(ref dL_dPNext[(offset + i) * 3 + 1], dP.G);
                Atomic.Add(ref dL_dPNext[(offset + i) * 3 + 2], dP.B);
            }
        }
    }

    protected static void ForwardKernal(Index2D index, ArrayView<Color> input, ArrayView<Color> convoluted, ArrayView<Color> kernal, ArrayView<GPUKernalFeatures> kF)
    {
        Color sum = new();

        for (int j = 0; j < kF[0].KernalSize; j++)
        {
            int offset = (index.Y * kF[0].Stride + j) * kF[0].InputWidth + index.X * kF[0].Stride;
            for (int i = 0; i < kF[0].KernalSize; i++)
            {
                sum += kernal[kF[0].KernalSize * j + i] * input[offset + i];
            }
        }
        convoluted[index.Y * kF[0].OutputWidth + index.X] = sum * kF[0].InverseKSquared;
    }
}