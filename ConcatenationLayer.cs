using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class ConcatenationLayer : Layer, IStructuralLayer
{
    private FeatureMap[,] _inputsSecondary;
    private FeatureMap[,] _outGradientsSecondary;

    private int _secondaryDimensions;

    public ConcatenationLayer() : base(1, 1) { }

    public override string Name => "Concatenation Layer";

    public void Connect(FeatureMap[,] inputs, FeatureMap[,] outGradients)
    {
        _inputsSecondary = inputs;
        _secondaryDimensions = inputs.GetLength(0);
        _batchSize = inputs.GetLength(1);
        _outGradientsSecondary = outGradients;

        for (int i = 0; i < _secondaryDimensions; i++)
        {
            int width = inputs[i, 0].Width;
            int length = inputs[i, 0].Length;
            for (int j = 0; j < _batchSize; j++)
            {
                _outGradientsSecondary[i, j] = new FeatureMap(width, length);
            }
        }
    }

    public override void Backwards(float learningRate)
    {
        for (int i = 0; i < _inputDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _outGradients[i,j] = _inGradients[i, j];
            }
        }

        for (int i = 0; i < _secondaryDimensions; i++)
        {
            for (int j = 0; j < _batchSize; j++)
            {
                _outGradientsSecondary[i, j] = _inGradients[_inputDimensions + i, j];
            }
        }
    }

    public override void Forward()
    {
        for(int i = 0; i < _inputDimensions; i++)
        {
            for(int j = 0; j < _batchSize; j++)
            {
                _outputs[i, j] = _inputs[i, j];
            }
        }

        for(int i = 0; i < _secondaryDimensions; i++)
        {
            for(int j = 0; j < _batchSize; j++)
            {
                _outputs[_inputDimensions + i, j] = _inputsSecondary[i, j];
            }
        }
    }

    public override void Reset()
    {
    }

    public override (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] inputs, FeatureMap[,] outGradients)
    {
        _inputDimensions = inputs.GetLength(0);
        _outputDimensions = _inputDimensions + _inputsSecondary.GetLength(0);

        _inputs = inputs;
        _outGradients = outGradients;
        _outputs = new FeatureMap[_outputDimensions, _batchSize];
        _inGradients = new FeatureMap[_outputDimensions, _batchSize];

        for (int i = 0; i < _inputDimensions; i++)
        {
            int width = _inputs[i, 0].Width;
            int length = _inputs[i, 0].Length;
            for (int j = 0; j < _batchSize; j++)
            {
                _outputs[i,j] = new FeatureMap(width,length);
                _outGradients[i, j] = new FeatureMap(width, length);
            }
        }

        for(int i = 0; i < _secondaryDimensions; i++)
        {
            int width = _inputsSecondary[i, 0].Width;
            int length = _inputsSecondary[i, 0].Length;
            for (int j = 0; j < _batchSize; j++)
            {
                _outputs[_inputDimensions + i, j] = new FeatureMap(width, length);
            }
        }

        return (_outputs, _inGradients);
    }
}
