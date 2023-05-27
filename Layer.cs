using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
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
    protected int _dimensions;

    protected FeatureMap[][] _outputs;
    protected FeatureMap[][] _outGradients;
    protected LayerInfo[] _layerInfos;

    public Layer(int kernalSize, int stride, ref FeatureMap[][] input)
    {
        _kernalSize = kernalSize;
        _stride = stride;
        _dimensions = input.Length;
        _layerInfos = new LayerInfo[_dimensions];
        _outputs = new FeatureMap[input.Length][];
        _outGradients = new FeatureMap[input.Length][];
        for(int i = 0; i < _dimensions; i++)
        {
            int batchSize = input[i].Length;
            _outputs[i] = new FeatureMap[batchSize];
            _outGradients[i] = new FeatureMap[batchSize];
            LayerInfo layer = _layerInfos[i] = new LayerInfo()
            {
                KernalSize = kernalSize,
                Stride = stride,
                InverseKSquared = 1f / (kernalSize * kernalSize),
                InputWidth = input[i][0].Width,
                InputLength = input[i][0].Length,
                OutputWidth = 2 + (input[i][0].Width - (kernalSize + stride)) / stride,
                OutputLength = 2 + (input[i][0].Length - (kernalSize + stride)) / stride
            };
            for (int j = 0; j < batchSize; j++)
            {
                _outGradients[i][j] = new FeatureMap(layer.InputWidth, layer.InputLength);
                _outputs[i][j] = new FeatureMap(layer.OutputWidth, layer.OutputLength);
            }
        }
        input = _outputs;
    }

    public abstract FeatureMap[][] Forward(FeatureMap[][] input);

    public abstract FeatureMap[][] Backwards(FeatureMap[][] input, FeatureMap[][] inGradient, float learningRate);
}

