using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.Layers.Loss;

public class WassersteinLoss : Loss
{


    private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, VariableView<float>, int> s_lossAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, VariableView<float>, int>(LossKernel);

    public override (float, float) GetLoss(Vector[] groundTruth)
    {
        var truth = Truth.GetArrayViewEmpty<float>();
        for (int i = 0; i < groundTruth.Length; i++)
        {
            groundTruth[i].CopyToBuffer(truth.SubView(i, 1));
        }

        Index1D index = new(groundTruth.Length);
        s_lossAction(index, Buffers.Output, truth, Losses.GetArrayViewZeroed<float>().VariableView(0), Accuracy.GetArrayViewZeroed<float>().VariableView(0), OutputShape.Volume);

        GPUManager.Accelerator.Synchronize();

        Truth.DecrementLiveCount();
        Losses.DecrementLiveCount();
        Accuracy.DecrementLiveCount();

        Losses.SyncCPU();
        Accuracy.SyncCPU();
        return (Losses[0] / groundTruth.Length, 0);
    }

    private static void LossKernel(Index1D index, ArrayView<float> output, ArrayView<float> truth, VariableView<float> totalLoss, VariableView<float> accuracy, int length)
    {
        int offset = index * length;
        float sum = 0;
        for (int i = 0; i < length; i++)
        {
            sum += output[offset + i];
        }

        float loss = MathF.Abs(sum) * truth[offset];

        for (int i = 0; i < length; i++)
        {
            output[offset + i] = sum * truth[offset];
        }

        Atomic.Add(ref totalLoss.Value, loss);
    }
}