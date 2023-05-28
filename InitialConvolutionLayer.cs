// See https://aka.ms/new-console-template for more information

using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.Cuda;

[Serializable]
public class InitialConvolutionLayer : ConvolutionalLayer
{
    public InitialConvolutionLayer(int kernalSize, int stride, ref FeatureMap[,] input) : base(kernalSize, stride, ref input)
    {
    }

    public FeatureMap[,] Forward(FeatureMap[] inputs)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using (Accelerator accelerator = context.CreateCudaAccelerator(0))
            {
                MemoryBuffer1D<Color, Stride1D.Dense>[,] deviceConvoluted = new MemoryBuffer1D<Color, Stride1D.Dense>[_inputDimensions, inputs.Length];
                
                for (int j = 0; j < inputs.Length; j++)
                {
                    using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = inputs[j].Allocate(accelerator);
                    
                    for (int i = 0; i < _inputDimensions; i++)
                    {

                        deviceConvoluted[i, j] = InitializeForwardKernal(i, deviceInput, accelerator);
                    }
                }
                
                accelerator.Synchronize();
                
                for (int i = 0; i < _inputDimensions; i++)
                {
                    for (int j = 0; j < inputs.Length; j++)
                    {
                        Convoluted[i,j].CopyFromBuffer(deviceConvoluted[i, j]);
                        deviceConvoluted[i, j].Dispose();
                    }
                }
            }
        }

        return Convoluted;
    }

    public FeatureMap[,] Backwards(FeatureMap[] inputs, FeatureMap[,] inGradients, float learningRate)
    {
        using (Context context = Context.Create(builder => builder.Cuda()))
        {
            using (Accelerator accelerator = context.CreateCudaAccelerator(0))
            {
                MemoryBuffer1D<float, Stride1D.Dense>[] deviceKernalGradient = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions];
                for (int j = 0; j < inputs.Length; j++)
                {
                    using MemoryBuffer1D<Color, Stride1D.Dense> deviceInput = inputs[j].Allocate(accelerator);
                    for (int i = 0; i < _inputDimensions; i++)
                    {
                        deviceKernalGradient[i] = accelerator.Allocate1D<float>(_kernalGradient[i].Length);
                        InitializeBackwardsKernal(i, deviceInput, inGradients[i,j], accelerator, deviceKernalGradient[i]);
                    }
                }
                
                accelerator.Synchronize();
                
                for (int i = 0; i < _inputDimensions; i++)
                {
                    deviceKernalGradient[i].CopyToCPU(_kernalGradient[i]);
                    deviceKernalGradient[i].Dispose();

                    for (int j = 0; j < _kernalSize * _kernalSize; j++)
                    {
                        _kernals[i][j] -= learningRate * LEARNINGMULTIPLIER * new Color(_kernalGradient[i][j * 3], _kernalGradient[i][j * 3 + 1], _kernalGradient[i][j * 3 + 2]).Clamp(CLAMP);
                    }
                }
            }
        }

        return _outGradients;
    }
}
