using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers
{
    public class Reshape : Layer, IStructuralLayer
    {
        Vector[] _proxy;

        [JsonProperty] private Shape[] OutputShapes { get => _outputShapes; set => _outputShapes = value; }

        public Reshape(Shape[] outputShapes) : base(1,1)
        {
            _outputShapes = outputShapes;
        }

        [JsonConstructor] private Reshape() { }

        public override string Name => "Reshape Layer";

        public override void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay)
        {
            if (_outputDimensions == 1)
            {
                for (int i = 0; i < _batchSize; i++)
                {
                    CopyFromArrayView(_buffers.InGradientsFloat[0, i], _buffers.OutGradientsFloat, _inputShapes, i);
                }
                Synchronize();
            }
            else if (_inputDimensions == 1)
            {
                for (int i = 0; i < _batchSize; i++)
                {
                    CopyToArrayView(_buffers.InGradientsFloat, _buffers.OutGradientsFloat[0, i], _outputShapes, i);
                }
                Synchronize();
            }
            else
            {
                for (int i = 0; i < _batchSize; i++)
                {
                    CopyToArrayView(_buffers.InGradientsFloat, _proxy[i].GetArrayView<float>(), _outputShapes, i);
                }
                Synchronize();

                for (int i = 0; i < _batchSize; i++)
                {
                    CopyFromArrayView(_proxy[i].GetArrayView<float>(), _buffers.OutGradientsFloat, _inputShapes, i);
                }
                Synchronize();

                for (int i = 0; i < _batchSize; i++)
                {
                    _proxy[i].DecrementLiveCount(2);
                }
            }
        }

        public override void Forward()
        {
            if (_inputDimensions == 1)
            {
                for (int i = 0; i < _batchSize; i++)
                {
                    CopyFromArrayView(_buffers.InputsFloat[0, i], _buffers.OutputsFloat, _outputShapes, i);
                }
                Synchronize();
            }
            else if(_outputDimensions == 1)
            {
                for(int i = 0; i < _batchSize; i++)
                {
                    CopyToArrayView(_buffers.InputsFloat, _buffers.OutputsFloat[0, i], _inputShapes, i);
                }
                Synchronize();
            }
            else
            {
                for(int i = 0; i < _batchSize; i++)
                {
                    CopyToArrayView(_buffers.InputsFloat, _proxy[i].GetArrayView<float>(), _inputShapes, i);
                }
                Synchronize();

                for(int i = 0; i < _batchSize; i++)
                {
                    CopyFromArrayView(_proxy[i].GetArrayView<float>(), _buffers.OutputsFloat, _outputShapes, i);
                }
                Synchronize();

                for(int i = 0; i < _batchSize; i++)
                {
                    _proxy[i].DecrementLiveCount(2);
                }
            }
        }

        public override void Reset()
        {
        }

        private static void CopyToArrayView(ArrayView<float>[,] inViews, ArrayView<float> outViews, Shape[] shapes, int batchIndex)
        {
            int start = 0;
            for (int i = 0; i < shapes.Length; i++)
            {
                int length = shapes[i].Area;
                Index1D index = new(length);
                GPUManager.CopyAction(index, inViews[i, batchIndex].SubView(0, length), outViews.SubView(start, length));
                start += length;
            }
        }

        private static void CopyFromArrayView(ArrayView<float> inViews, ArrayView<float>[,] outBuffer, Shape[] outViews, int batchIndex)
        {
            int start = 0;
            for (int i = 0; i < outViews.Length; i++)
            {
                int length = outViews[i].Area;
                Index1D index = new(length);
                GPUManager.CopyAction(index, inViews.SubView(start, length), outBuffer[i, batchIndex].SubView(0, length));
                start += length;
            }
        }

        public override Shape[] Startup(Shape[] inputShapes, IOBuffers buffers, uint batchSize)
        {

            _inputDimensions = inputShapes.Length;
            _outputDimensions = _outputShapes.Length;

            _batchSize = batchSize;
            _layerInfos = new ILayerInfo[_inputDimensions];
            _inputShapes = inputShapes;

            int inputLength = 0;
            for(int i = 0; i < _inputDimensions; i++)
            {
                inputLength += inputShapes[i].Area;
            }

            int outputLength = 0;
            for(int i = 0; i < _outputDimensions; i++)
            {
                outputLength += _outputShapes[i].Area;
            }

            if(inputLength != outputLength)
            {
                throw new ArgumentException("Input and output shapes have different lengths.");
            }

            if(_inputDimensions != 1)
            {
                _proxy = new Vector[batchSize];
                for (int i = 0; i < batchSize; i++)
                {
                    _proxy[i] = new Vector(inputLength);
                }
            }

            _buffers = buffers;
            for (int i = 0; i < _outputDimensions; i++)
                buffers.OutputDimensionArea(i, _outputShapes[i].Area);

            return _outputShapes;
        }
    }
}
