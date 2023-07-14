using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers
{

    /// <summary>
    /// The <see cref="Summation"/> class is a <see cref="Layer"/> that sums the <see cref="FeatureMap"/>s across multiple dimensions, reducing the number of
    /// dimensions in the <see cref="Network"/>.
    /// Note: When summing <see cref="FeatureMap"/>s that have been batch normalized to have a mean of 0, the mean will remain the same, but the standard deviation
    /// will change.
    /// </summary>
    [Serializable]
    public class Summation : Layer, IStructuralLayer
    {
        private readonly Action<Index1D, ArrayView<float>, ArrayView<float>> s_forwardAction = GPU.GPUManager.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(ForwardKernal);
        [JsonProperty] private int _dimensionDivisor;

        [JsonConstructor] public Summation() : base(1, 1) { }

        /// <inheritdoc/>
        public override string Name => "Summation Layer";
        /// <inheritdoc/>
        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            for (int i = 0; i < _inputDimensions; i++)
            {
                int outputDimension = i % _outputDimensions;
                Index1D index = new(Infos(i).Area);
                for (int j = 0; j < _batchSize; j++)
                {
                    GPU.GPUManager.CopyAction(index, _buffers.InGradientsColor[outputDimension, j], _buffers.OutGradientsColor[i, j]);
                }
            }

            Synchronize();
        }

        /// <inheritdoc/>
        public override void Forward()
        {
            for(int i = 0; i < _outputDimensions; i++)
            {
                for (int j = 0; j < _batchSize; j++)
                {
                    _buffers.OutputsColor[i, j].SubView(0, Infos(i).Area).MemSetToZero();
                }
            }

            for (int i = 0; i < _inputDimensions; i++)
            {
                int outputDimension = i % _outputDimensions;
                Index1D index = new(Infos(i).Area * 3);
                for (int j = 0; j < _batchSize; j++)
                {
                    s_forwardAction(index, _buffers.InputsFloat[i, j], _buffers.OutputsFloat[outputDimension, j]);
                }
            }

            Synchronize();
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <summary>
        /// Sets the number of dimensions for the output. The input dimensions will need to be a multiple of the output dimensions
        /// or vice versa. Overwrites <see cref="SetOutputDivisor(int)"/>.
        /// </summary>
        /// <param name="dimensions">The number of output dimensions.</param>
        public void SetOutputDimensions(int dimensions)
        {
            _outputDimensions = dimensions;
        }

        /// <summary>
        /// Sets the number of dimensions for the output as a multiple of the input dimensions.
        /// Is Overwritten by <see cref="SetOutputDimensions(int)"/>.
        /// </summary>
        /// <param name="divisor">The factor to multiply the input dimensions to set the output dimensions.
        /// Must be greater than 1.</param>
        public void SetOutputDivisor(int divisor)
        {
            if(divisor <= 1)
            {
                throw new ArgumentException("Dimension divisor must be greater than 1.");
            }
            _dimensionDivisor = divisor;
        }

        /// <inheritdoc/>
        public override FeatureMap[,] Startup(FeatureMap[,] inputs, IOBuffers buffers)
        {
            if(_outputDimensions != 0)
            {
                _dimensionDivisor = inputs.GetLength(0) / _outputDimensions;
            }
            
            BaseStartup(inputs, buffers, -_dimensionDivisor);

            return _outputs;
        }

        /// <summary>
        /// An <see cref="ILGPU"/> kernal that adds the values from one <see cref="ArrayView{T}"/> of floats, to another.
        /// </summary>
        /// <param name="index">The index of the arrays to sum.</param>
        /// <param name="input">The array of floats being added to <paramref name="output"/>.</param>
        /// <param name="output">The array of floats to which <paramref name="input"/> is being added.</param>
        private static void ForwardKernal(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            Atomic.Add(ref output[index.X], input[index.X]);
        }

        /// <summary>
        /// Gets the <see cref="StaticLayerInfo"/> for a particular dimension.
        /// </summary>
        /// <param name="index">The dimension who <see cref="StaticLayerInfo"/> is needed.</param>
        /// <returns>Return the <see cref="StaticLayerInfo"/> corresponding to an input dimension.</returns>
        private StaticLayerInfo Infos(int index)
        {
            return (StaticLayerInfo)_layerInfos[index];
        }
    }
}
