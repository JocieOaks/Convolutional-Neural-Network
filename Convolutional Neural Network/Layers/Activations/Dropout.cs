﻿using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;

 namespace ConvolutionalNeuralNetwork.Layers.Activations
{
    /// <summary>
    /// The <see cref="Dropout"/> class is a <see cref="Layer"/> that adds noise to a <see cref="Tensor"/> by randomly
    /// dropping values from the <see cref="Tensor"/>.
    /// </summary>
    public class Dropout : Layer
    {
        private static readonly Action<Index2D, ArrayView<float>, ArrayView<byte>> s_forwardAction = GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<byte>>(ForwardKernel);
        private readonly float _dropoutRate;
        private ByteArray _dropoutValues;

        /// <summary>
        /// Initializes a new instance of the <see cref="Dropout"/> class.
        /// </summary>
        /// <param name="dropoutRate">The frequency in which values are dropped.</param>
        public Dropout(float dropoutRate) : base(1, 1)
        {
            _dropoutRate = dropoutRate > 0 ? dropoutRate : 0.2f;
        }

        /// <inheritdoc/>
        public override string Name => "Dropout Layer";

        /// <inheritdoc />
        public override bool Reflexive => true;
        /// <inheritdoc/>
        public override void Backwards(int batchSize, bool update)
        {
            ApplyDropout(batchSize * InputShape.Volume, false);
        }
        /// <inheritdoc/>
        public override void Forward(int batchSize)
        {
            ApplyDropout(batchSize * InputShape.Volume, true);
        }

        /// <inheritdoc/>
        public override TensorShape Startup(TensorShape inputShape, PairedGPUViews views, int maxBatchSize)
        {
            if (Initialized)
                return OutputShape;
            BaseStartup(inputShape, views);
            int byteLength = inputShape.Volume * maxBatchSize / 8 + (inputShape.Volume * maxBatchSize % 8 == 0 ? 0 : 1);
            _dropoutValues = new ByteArray(byteLength);

            return OutputShape;
        }

        private static void ForwardKernel(Index2D index, ArrayView<float> input, ArrayView<byte> dropout)
        {
            int inputIndex = 8 * index.X + index.Y;
            input[inputIndex] = (dropout[index.X] & (1 << index.Y - 1)) == 0 ? 0 : input[inputIndex];
        }

        private void ApplyDropout(int length, bool randomizeDropout)
        {
            int byteLength = length / 8;
            int tail = length % 8;
            if (randomizeDropout)
            {
                for (int i = 0; i < byteLength; i++)
                {
                    byte dropoutByte = byte.MaxValue;
                    for (int j = 0; j < 8; j++)
                    {
                        if (Utility.Random.NextDouble() < _dropoutRate)
                            dropoutByte &= (byte)~(1 << j);
                    }

                    _dropoutValues[i] = dropoutByte;
                }


                if (tail > 0)
                {
                    //Remaining values after the tail remain as 1 bits, so they are left unaffected.
                    byte dropoutByte = byte.MaxValue;
                    for (int j = 0; j < tail; j++)
                    {
                        if (Utility.Random.NextDouble() < _dropoutRate)
                            dropoutByte &= (byte)~(1 << j);
                    }

                    _dropoutValues[byteLength] = dropoutByte;
                }
                _dropoutValues.CopyToView();
            }

            if (tail > 0)
            {
                byteLength += 1;
            }

            Index2D index = new(byteLength, 8);
            
            s_forwardAction(index, Views.Input, _dropoutValues.GetArrayView());
            GPUManager.Accelerator.Synchronize();
            _dropoutValues.Release();
        }
    }
}