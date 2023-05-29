using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

public abstract class Layer
{
    [JsonProperty]
    protected int _kernalSize;
    [JsonProperty]
    protected int _stride;
    [JsonProperty]
    protected int _inputDimensions;
    [JsonProperty]
    protected int _outputDimensions;
    protected int _batchSize;

    protected FeatureMap[,] _outputs;
    protected FeatureMap[,] _outGradients;
    protected ILayerInfo[] _layerInfos;

    public abstract string Name { get; }

    public Layer(int kernalSize, int stride, ref FeatureMap[,] input, int outputDimensionFactor = 1)
    {
        _kernalSize = kernalSize;
        _stride = stride;
        _inputDimensions = input.GetLength(0);
        if(outputDimensionFactor >= 1)
        {
            _outputDimensions = outputDimensionFactor * _inputDimensions;
        }
        else
        {
            if(outputDimensionFactor == 0 || _inputDimensions % outputDimensionFactor != 0)
            {
                throw new ArgumentException("outputDimensionFactor does not divide evenly with input dimensions.");
            }
            else
            {
                _outputDimensions = _inputDimensions / -outputDimensionFactor;
            }
        }

        _batchSize = input.GetLength(1);
        _layerInfos = new ILayerInfo[_inputDimensions];
        _outGradients = new FeatureMap[_inputDimensions, _batchSize];
        _outputs = new FeatureMap[_outputDimensions, _batchSize];
        for(int i = 0; i < _inputDimensions; i++)
        {
            ILayerInfo layer;
            if (stride == 1 && kernalSize == 1)
            {
                layer = _layerInfos[i] = new SingleLayerInfo()
                {
                    Width = input[i, 0].Width,
                    Length = input[i, 1].Length,
                };
            }
            else
            {
                layer = _layerInfos[i] = new LayerInfo()
                {
                    KernalSize = kernalSize,
                    Stride = stride,
                    InverseKSquared = 1f / (kernalSize * kernalSize),
                    InputWidth = input[i, 0].Width,
                    InputLength = input[i, 0].Length,
                    OutputWidth = 2 + (input[i, 0].Width - kernalSize - 1) / stride,
                    OutputLength = 2 + (input[i, 0].Length - kernalSize - 1) / stride
                };
            }

            for (int j = 0; j < _batchSize; j++)
            {
                _outGradients[i, j] = new FeatureMap(layer.InputWidth, layer.InputLength);
            }

        }
        for(int i = 0; i < _outputDimensions; i++)
        {
            ILayerInfo layer;
            if(outputDimensionFactor >= 1)
            {
                layer = _layerInfos[i / outputDimensionFactor];
            }
            else
            {
                layer = _layerInfos[i * -outputDimensionFactor];
            }

            for (int j = 0; j < _batchSize; j++)
            {
                _outputs[i, j] = new FeatureMap(layer.OutputWidth, layer.OutputLength);
            }
        }
        input = _outputs;
    }

    public abstract FeatureMap[,] Forward(FeatureMap[,] input);

    public abstract FeatureMap[,] Backwards(FeatureMap[,] input, FeatureMap[,] inGradient, float learningRate);
}

