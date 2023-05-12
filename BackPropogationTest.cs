using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;


public class BackPropogationTest
{

    readonly FeatureMap[][] _initialInput;
    FeatureMap[][] _intermediate;
    Vector _finalOutput;

    VectorizationLayer _testLayer;
    Layer _impactLayer;

    public BackPropogationTest()
    {
        _initialInput = new FeatureMap[1][];
        _initialInput[0] = new FeatureMap[1];
        

        for (int k = 0; k < 1; k++)
        {
            _initialInput[0][k] = new FeatureMap(20, 20);
            for (int i = 0; i < 20; i++)
            {
                for (int j = 0; j < 20; j++)
                {
                    _initialInput[0][0][i, j] = Color.Random(1);
                }
            }
        }

        _testLayer = new VectorizationLayer(5, 1);
        _impactLayer = new ConvolutionalLayer(1, 3, 1);
    }

    public float Test(float testAlpha, float propAlpha)
    {
        Forward();
        float loss = Loss();
        Vector dL_dP = Gradient(loss);
        Backward(dL_dP, testAlpha, propAlpha);
        return loss;
    }

    public float Loss()
    {
        float sum = 0;
        for(int i = 0; i < _finalOutput.Length; i++)
        {
            sum += MathF.Pow(_finalOutput[i], 2);
        }

        return sum;
    }

    public Vector Gradient(float loss)
    {
        Vector dL_dP = _finalOutput * 2;
        /*FeatureMap[][] dL_dP = new FeatureMap[_finalOutput.Length][];
        for (int i = 0; i < _finalOutput.Length; i++)
        {
            dL_dP[i] = new FeatureMap[_finalOutput[i].Length];
            for (int j = 0; j < _finalOutput[i].Length; j++)
            {
                dL_dP[i][j] = new FeatureMap(_finalOutput[i][j].Width, _finalOutput[i][j].Length);
                for (int k = 0; k < dL_dP[i][j].Width; k++)
                {
                    for (int l = 0; l < dL_dP[i][j].Length; l++)
                    {
                        if (_finalOutput[i][j][k, l].SquareMagnitude == 0)
                            dL_dP[i][j][k, l] = new Color();
                        else
                            dL_dP[i][j][k, l] = (2 / _finalOutput[i][j][k, l].Magnitude / _finalOutput[i][j].Area) * _finalOutput[i][j][k, l];
                    }
                }
            }
        }*/
        return dL_dP;
    }

    public void Forward()
    {
        _intermediate = _impactLayer.Forward(_initialInput);
        _finalOutput = _testLayer.Forward(_intermediate[0]);
    }

    public void Backward(Vector gradient, float testAlpha, float propAlpha)
    {
        FeatureMap[][] dL_dP = new FeatureMap[1][];
        dL_dP[0] = _testLayer.Backwards(gradient, _finalOutput, _intermediate[0], testAlpha);
        _impactLayer.Backwards(dL_dP, _initialInput, propAlpha);
    }
}

