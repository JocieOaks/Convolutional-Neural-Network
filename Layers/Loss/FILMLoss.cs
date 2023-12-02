using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace ConvolutionalNeuralNetwork.Layers.Loss
{
    public class FILMLoss : Loss
    {
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, int> s_lossAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, int>(LossKernel);

        public override (float, float)  GetLoss(Vector[] groundTruth)
        {
            var truth = Truth.GetArrayViewEmpty<float>();
            for(int i = 0; i < groundTruth.Length; i++)
            {
                groundTruth[i].CopyToBuffer(truth.SubView(i * OutputShape.Volume, OutputShape.Volume));
            }

            Index1D index = new(groundTruth.Length);
            s_lossAction(index, Buffers.Output, truth, Losses.GetArrayViewZeroed<float>().VariableView(0), OutputShape.Volume);

            GPUManager.Accelerator.Synchronize();

            Truth.DecrementLiveCount();
            Losses.DecrementLiveCount();

            Losses.SyncCPU();
            return (Losses[0], 1);
        }

        private static void LossKernel(Index1D index, ArrayView<float> output, ArrayView<float> truth, VariableView<float> totalLoss, int length)
        {
            int offset = index * length;
            float loss = 0;

            for(int i = 0; i < length; i++)
            {
                float defect = output[offset + i] - truth[offset + i];
                loss += XMath.Abs(defect);
                output[offset + i] = XMath.Sign(defect);
            }

            for(int i = 0; i < length; i++)
            {
                output[offset + i] *= loss;
            }

            Atomic.Add(ref totalLoss.Value, loss);
        }
    }
}
