using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU.Runtime;
using ILGPU;
using ILGPU.Algorithms;

namespace ConvolutionalNeuralNetwork.Layers.Loss
{
    public class CrossEntropyLoss : Loss
    {


        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, VariableView<float>, int> s_lossAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, VariableView<float>, int>(LossKernel);

        public override (float, float) GetLoss(Vector[] groundTruth)
        {
            var truth = Truth.GetArrayViewEmpty();
            for (int i = 0; i < groundTruth.Length; i++)
            {
                groundTruth[i].CopyToBuffer(truth.SubView(i * OutputShape.Volume, OutputShape.Volume));
            }

            Index1D index = new(groundTruth.Length);
            s_lossAction(index, views.Output, truth, Losses.GetArrayViewZeroed().VariableView(0), Accuracy.GetArrayViewZeroed().VariableView(0), OutputShape.Volume);

            GPUManager.Accelerator.Synchronize();

            Truth.Release();
            Losses.Release();
            Accuracy.Release();

            Losses.SyncCPU();
            Accuracy.SyncCPU();
            return (Losses[0] / groundTruth.Length, Accuracy[0] / groundTruth.Length);
        }

        private static void LossKernel(Index1D index, ArrayView<float> output, ArrayView<float> truth, VariableView<float> totalLoss, VariableView<float> accuracy, int length)
        {
            int offset = index * length;
            float product = 0;
            for (int i = 0; i < length; i++)
            {
                product += output[offset + i] * truth[offset + i];
            }

            float score = (product + 1) / 2;

            float loss = -XMath.Log(score + Utility.ASYMPTOTE_ERROR_CORRECTION);
            Atomic.Add(ref accuracy.Value, XMath.Round(score));

            for (int i = 0; i < length; i++)
            {
                output[offset + i] = (-1 / (score + Utility.ASYMPTOTE_ERROR_CORRECTION)) * truth[offset + i];
            }

            Atomic.Add(ref totalLoss.Value, loss);
        }
    }
}
