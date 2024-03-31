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
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, VariableView<float>, VariableView<float>, int> s_multiclassLossAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, VariableView<float>, VariableView<float>, int>(MulticlassLossKernel);

        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, VariableView<float>> s_lossAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, VariableView<float>>(SingleClassLossKernel);

        /// <inheritdoc />
        public override (float, float) GetLoss(Vector[] labels, Vector classifications)
        {
            ArrayView<float> labelView = Labels.GetArrayViewEmpty();
            ArrayView<float> classificationView = Classifications.GetArrayViewEmpty();
            int batchSize = classifications.Length;

            
            classifications.CopyToView(classificationView);

            Index1D index = new(batchSize);
            if (OutputShape.Volume > 1)
            {
                for (int i = 0; i < batchSize; i++)
                {

                    labels[i].CopyToView(labelView.SubView(i * OutputShape.Volume, OutputShape.Volume));

                }
                s_multiclassLossAction(index, Views.Output, labelView, classificationView, Losses.GetArrayViewZeroed().VariableView(0), 
                    Accuracy.GetArrayViewZeroed().VariableView(0), OutputShape.Volume);
            }
            else
            {
                s_lossAction(index, Views.Output, classificationView, Losses.GetArrayViewZeroed().VariableView(0),
                    Accuracy.GetArrayViewZeroed().VariableView(0));
            }

            GPUManager.Accelerator.Synchronize();

            Labels.Release();
            Losses.Release();
            Accuracy.Release();
            Classifications.Release();

            Losses.SyncCPU();
            Accuracy.SyncCPU();
            return (Losses[0] / classifications.Length, Accuracy[0] / classifications.Length);
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

        private static void MulticlassLossKernel(Index1D index, ArrayView<float> output, ArrayView<float> labels, ArrayView<float> classification, VariableView<float> totalLoss, VariableView<float> accuracy, int length)
        {
            int offset = index * length;
            float loss = 0;
            float maxProbability = 0;
            float maxLabel = 0;

            for (int i = 0; i < length; i++)
            {
                loss += -XMath.Log(output[offset + i] + Utility.ASYMPTOTE_ERROR_CORRECTION) * labels[offset + i];
                if (output[offset + i] > maxProbability)
                {
                    maxProbability = output[offset + i];
                    maxLabel = labels[offset + i];
                }
            }
            Atomic.Add(ref accuracy.Value, maxLabel);

            for (int i = 0; i < length; i++)
            {
                output[offset + i] = (2 * classification[index] - 1) * loss * -labels[offset + i] / (output[offset + i] + Utility.ASYMPTOTE_ERROR_CORRECTION);
            }

            Atomic.Add(ref totalLoss.Value, loss);
        }
    }
}
