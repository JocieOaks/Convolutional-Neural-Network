using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers.Loss
{
    public class FILMLoss : Loss
    {
        private static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, int> s_lossAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, VariableView<float>, int>(LossKernel);

        public override float GetLoss(Vector[] groundTruth)
        {
            var truth = _truth.GetArrayViewEmpty<float>();
            for(int i = 0; i < groundTruth.Length; i++)
            {
                groundTruth[i].CopyToBuffer(truth.SubView(i * _outputShape.Volume, _outputShape.Volume));
            }

            Index1D index = new(groundTruth.Length);
            s_lossAction(index, _buffers.Output, truth, _loss.GetArrayViewZeroed<float>().VariableView(0), _outputShape.Volume);

            GPUManager.Accelerator.Synchronize();

            _truth.DecrementLiveCount();
            _loss.DecrementLiveCount();

            _loss.SyncCPU();
            return _loss[0];
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
