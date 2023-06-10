using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime.Cuda;
using Newtonsoft.Json;

public class SkipConnectionLayer : Layer, IStructuralLayer
{
    public override string Name => "Skip Connection Layer";

    FeatureMap[,] _inGradientSecondary;
    [JsonProperty] ConcatenationLayer _concatenationLayer;
    MemoryBuffer1D<Color, Stride1D.Dense>[,] _deviceInGradientsSecondary;

    public SkipConnectionLayer() : base(1, 1) { }

    public ConcatenationLayer GetConcatenationLayer()
    {
        _concatenationLayer = new ConcatenationLayer();
        return _concatenationLayer;
    }

    public override void Backwards(float learningRate)
    {
Context context = ConvolutionalNeuralNetwork.Context;
        using Accelerator accelerator = context.CreateCudaAccelerator(0);

        var backwardsKernal = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<Color>, ArrayView<Color>, ArrayView<float>>(BackwardsKernal);

        for(int i = 0; i < _inputDimensions; i++)
        {
            Index2D index = new Index2D(Infos(i).Area, 3);
            for(int j = 0; j < _batchSize; j++)
            {
                _deviceInGradients[i, j] = _inGradients[i, j].Allocate(accelerator);
                _deviceInGradientsSecondary[i, j] = _inGradientSecondary[i, j].Allocate(accelerator);
                _deviceOutGradients[i, j] = _outGradients[i, j].AllocateFloat(accelerator);

                backwardsKernal(index, _deviceInGradients[i, j].View, _deviceInGradientsSecondary[i, j].View, _deviceOutGradients[i, j].View);
            }
        }

        accelerator.Synchronize();

        for(int i = 0; i< _inputDimensions; i++)
        {
            for(int j = 0; j< _batchSize; j++)
            {
                _outGradients[i, j].CopyFromBuffer(_deviceOutGradients[i, j]);
                _deviceOutGradients[i, j].Dispose();
                _deviceInGradients[i, j].Dispose();
                _deviceInGradientsSecondary[i, j].Dispose();
            }
        }
    }

    public override void Forward()
    {
    }

    public override void Reset()
    {
    }

    public override (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] inputs, FeatureMap[,] outGradients)
    {

        _outputDimensions = _inputDimensions = inputs.GetLength(0);

        _batchSize = inputs.GetLength(1);
        _layerInfos = new ILayerInfo[_inputDimensions];
        _inputs = _outputs = inputs;

        _outGradients = outGradients;

        _inGradients = new FeatureMap[_outputDimensions, _batchSize];
        _inGradientSecondary = new FeatureMap[_outputDimensions, _batchSize];

        for (int i = 0; i < _inputDimensions; i++)
        {
            ILayerInfo layer;
            layer = _layerInfos[i] = new SingleLayerInfo()
            {
                Width = inputs[i, 0].Width,
                Length = inputs[i, 0].Length,
            };


            for (int j = 0; j < _batchSize; j++)
            {
                _outGradients[i, j] = new FeatureMap(layer.InputWidth, layer.InputLength);
            }
        }

        _deviceInGradients = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions, _batchSize];
        _deviceOutGradients = new MemoryBuffer1D<float, Stride1D.Dense>[_inputDimensions, _batchSize];
        _deviceInGradientsSecondary = new MemoryBuffer1D<Color, Stride1D.Dense>[_outputDimensions, _batchSize];

        _concatenationLayer.Connect(inputs, _inGradientSecondary);

        return (inputs, _inGradients);
    }

    private static void BackwardsKernal(Index2D index, ArrayView<Color> inGradient1, ArrayView<Color> inGradient2, ArrayView<float> outGradient)
    {
        outGradient[index.X * 3 + index.Y] = inGradient1[index.X][index.Y] + inGradient2[index.X][index.Y];
    }

    private SingleLayerInfo Infos(int index)
    {
        return (SingleLayerInfo)_layerInfos[index];
    }
}
