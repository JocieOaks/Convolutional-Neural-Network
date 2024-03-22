using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU.Runtime;
using ILGPU;
using ILGPU.Algorithms;

namespace ConvolutionalNeuralNetwork.Layers.Loss
{
    /// <summary>
    /// The <see cref="CrossEntropyLoss"/> class determines the loss of the <see cref="Network"/> using cross entropy loss.
    /// </summary>
    public class CrossEntropyLoss : Loss
    {
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, VariableView<float>, int> s_multiclassLossAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, VariableView<float>, int>(MulticlassLossKernel);

        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, VariableView<float>> s_lossAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, VariableView<float>>(SingleClassLossKernel);

        /// <inheritdoc />
        public override (float, float) GetLoss(Vector[] groundTruth)
        {
            ArrayView<float> truth = Truth.GetArrayViewEmpty();
            for (int i = 0; i < groundTruth.Length; i++)
            {
                groundTruth[i].CopyToBuffer(truth.SubView(i * OutputShape.Volume, OutputShape.Volume));
            }

            Index1D index = new(groundTruth.Length);
            if (OutputShape.Volume > 1)
            {
                s_multiclassLossAction(index, Views.Output, truth, Losses.GetArrayViewZeroed().VariableView(0), 
                    Accuracy.GetArrayViewZeroed().VariableView(0), OutputShape.Volume);
            }
            else
            {
                s_lossAction(index, Views.Output, truth, Losses.GetArrayViewZeroed().VariableView(0),
                    Accuracy.GetArrayViewZeroed().VariableView(0));
            }

            GPUManager.Accelerator.Synchronize();

            Truth.Release();
            Losses.Release();
            Accuracy.Release();

            Losses.SyncCPU();
            Accuracy.SyncCPU();
            return (Losses[0] / groundTruth.Length, Accuracy[0] / (groundTruth.Length * OutputShape.Volume));
        }

        private static void SingleClassLossKernel(Index1D index, ArrayView<float> output, ArrayView<float> classification, VariableView<float> totalLoss, VariableView<float> accuracy)
        {
            float truth = classification[index];
            float inverse = 1 - classification[index];
            float probability = output[index];
            float inverseProbability = 1 - output[index];

            float loss = -(truth * XMath.Log(probability) + inverse * XMath.Log(inverseProbability));
            Atomic.Add(ref accuracy.Value, truth * XMath.Round(output[index]) + inverse * XMath.Round(inverseProbability));

            output[index] = loss * (-truth / (probability + Utility.ASYMPTOTE_ERROR_CORRECTION) + inverse / (inverseProbability + Utility.ASYMPTOTE_ERROR_CORRECTION));

            Atomic.Add(ref totalLoss.Value, loss);
        }

        private static void MulticlassLossKernel(Index1D index, ArrayView<float> output, ArrayView<float> classification, VariableView<float> totalLoss, VariableView<float> accuracy, int length)
        {
            int offset = index * length;
            float loss = 0;
            for (int i = 0; i < length; i++)
            {
                loss += -XMath.Log(output[offset + i] + Utility.ASYMPTOTE_ERROR_CORRECTION) * classification[offset + i];
                Atomic.Add(ref accuracy.Value, XMath.Round(output[offset + i]));
            }

            for (int i = 0; i < length; i++)
            {
                output[offset + i] = loss * (-classification[offset + i] / (output[offset + i] + Utility.ASYMPTOTE_ERROR_CORRECTION));
            }

            Atomic.Add(ref totalLoss.Value, loss);
        }
    }
}
