using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Algorithms;

namespace ConvolutionalNeuralNetwork.Layers.Loss
{
    public class CrossEntropyLoss : Loss
    {


        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, VariableView<float>, int> s_lossAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, VariableView<float>, int>(LossKernel);

        public override (float, float) GetLoss(Vector[] groundTruth)
        {
            var truth = _truth.GetArrayViewEmpty<float>();
            for (int i = 0; i < groundTruth.Length; i++)
            {
                groundTruth[i].CopyToBuffer(truth.SubView(i * _outputShape.Volume, _outputShape.Volume));
            }

            Index1D index = new(groundTruth.Length);
            s_lossAction(index, _buffers.Output, truth, _loss.GetArrayViewZeroed<float>().VariableView(0), _accuracy.GetArrayViewZeroed<float>().VariableView(0), _outputShape.Volume);

            GPUManager.Accelerator.Synchronize();

            _truth.DecrementLiveCount();
            _loss.DecrementLiveCount();
            _accuracy.DecrementLiveCount();

            _loss.SyncCPU();
            _accuracy.SyncCPU();
            return (_loss[0] / groundTruth.Length, _accuracy[0] / groundTruth.Length);
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

            float loss = -XMath.Log(score + Utility.ASYMPTOTEERRORCORRECTION);
            Atomic.Add(ref accuracy.Value, XMath.Round(score));

            for (int i = 0; i < length; i++)
            {
                output[offset + i] = (-1 / (score + Utility.ASYMPTOTEERRORCORRECTION)) * truth[offset + i];
            }

            Atomic.Add(ref totalLoss.Value, loss);
        }
    }
}
