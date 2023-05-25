// See https://aka.ms/new-console-template for more information

using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.Cuda;

[Serializable]
public class InitialConvolutionLayer : ConvolutionalLayer
{
    public InitialConvolutionLayer(int kernalSize, int stride, ref FeatureMap[][] input) : base(kernalSize, stride, ref input)
    {
    }

    public FeatureMap[][] Forward(FeatureMap[] input)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using (Accelerator accelerator = context.CreateCudaAccelerator(0))
            {
                MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceConvoluted = new MemoryBuffer1D<Color, Stride1D.Dense>[_dimensions, input.Length];
                for (int j = 0; j < input.Length; j++)
                {
                    using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = input[j].Allocate(accelerator);
                    for (int i = 0; i < _dimensions; i++)
                    {
                        deviceConvoluted[i, j] = _convoluted[i][j].AllocateEmpty(accelerator);
                        using MemoryBuffer1D<Color, Stride1D.Dense> deviceKernal = accelerator.Allocate1D(_kernals[i]);
                        using MemoryBuffer1D<KernalFeatures, Stride1D.Dense> deviceKernalFeatures = accelerator.Allocate1D<KernalFeatures>(new KernalFeatures[] { new KernalFeatures(input[j].Width, _convoluted[i][j].Width, _kernalSize, _stride, _invK2) });
                        using AcceleratorStream stream = accelerator.CreateStream();
                        Action<AcceleratorStream, Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<KernalFeatures>> forwardKernal =
                            accelerator.LoadAutoGroupedKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<KernalFeatures>>(ForwardKernal);
                        Index2D index = new(_convoluted[i][j].Width, _convoluted[i][j].Length);
                        forwardKernal(stream, index, deviceInput.View, deviceConvoluted[i, j].View, deviceKernal.View, deviceKernalFeatures.View);
                    }
                }
                accelerator.Synchronize();
                for (int i = 0; i < _dimensions; i++)
                {
                    for (int j = 0; j < input.Length; j++)
                    {
                        _convoluted[i][j].CopyFromBuffer(deviceConvoluted[i, j]);
                        deviceConvoluted[i, j].Dispose();
                    }
                }
            }
        }

        return _convoluted;
    }

    public FeatureMap[][] Backwards(FeatureMap[] input, FeatureMap[][] dL_dP, float learningRate)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using (Accelerator accelerator = context.CreateCudaAccelerator(0))
            {
                MemoryBuffer1D<float, Stride1D.Dense>[] deviceDL_dK = new MemoryBuffer1D<float, Stride1D.Dense>[_dimensions];
                for (int j = 0; j < input.Length; j++)
                {
                    using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = input[j].Allocate(accelerator);
                    for (int i = 0; i < _dimensions; i++)
                    {
                        deviceDL_dK[i] = accelerator.Allocate1D<float>(_dL_dK[i].Length);
                        using MemoryBuffer1D<float, Stride1D.Dense> deviceDL_dPNext = accelerator.Allocate1D<float>(input[j].Area * 3);
                        using MemoryBuffer1D<Color, Stride1D.Dense> deviceKernal = accelerator.Allocate1D(_kernals[i]);
                        using MemoryBuffer1D<Color, Stride1D.Dense> deviceDL_dP = dL_dP[i][j].Allocate(accelerator);
                        using MemoryBuffer1D<KernalFeatures, Stride1D.Dense> deviceKernalFeatures = accelerator.Allocate1D<KernalFeatures>(new KernalFeatures[] { new KernalFeatures(input[j].Width, dL_dP[i][j].Width, _kernalSize, _stride, _invK2) });
                        using AcceleratorStream stream = accelerator.CreateStream();

                        Action<AcceleratorStream, Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<float>, ArrayView<KernalFeatures>> backWardsKernal =
                            accelerator.LoadAutoGroupedKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<Color>, ArrayView<float>, ArrayView<float>, ArrayView<KernalFeatures>>(BackwardsKernal);

                        Index2D index = new(dL_dP[i][j].Width, dL_dP[i][j].Length);

                        backWardsKernal(stream, index, deviceInput.View, deviceKernal.View, deviceDL_dP.View, deviceDL_dPNext.View, deviceDL_dK[i].View, deviceKernalFeatures.View);

                    }
                }
                accelerator.Synchronize();
                for (int i = 0; i < _dimensions; i++)
                {
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
}
