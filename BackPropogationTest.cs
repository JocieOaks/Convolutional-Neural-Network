using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;


public class BackPropogationTest
{

    readonly FeatureMap[][] _initialInput;
    FeatureMap[][] _oldPropagation;
    FeatureMap[][] _newPropagation;

    ConvolutionalLayerGPU _layerGPU;
    ConvolutionalLayer _layer;

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
                    _initialInput[0][0][i, j] = new Color(i % 4 * 0.25f, j % 4 * 0.25f, -i % 4 * 0.25f);
                }
            }
        }

        FeatureMap[][] current = _initialInput;
        _layerGPU = new ConvolutionalLayerGPU(3, 1, ref current);
        current = _initialInput;
        _layer = new ConvolutionalLayer(3, 1, ref current);
    }

    public float Test(float testLearningRate, float propLearningRate)
    {
        Forward();
        float loss = Loss();
        Vector dL_dP = Gradient(loss);
        Backward(dL_dP, testLearningRate, propLearningRate);
        return loss;
    }

    public float Loss()
    {
        float sum = 0;

        return sum;
    }

    public Vector Gradient(float loss)
    {
        Vector dL_dP = new Vector(4)* 2;
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
        _oldPropagation = _layer.Forward(_initialInput);
        _newPropagation = _layerGPU.Forward(_initialInput);

        for(int i = 0; i < _oldPropagation[0][0].Width; i++)
        {
            for(int j = 0; j < _oldPropagation[0][0].Length; j++)
            {
                Color old = _oldPropagation[0][0][j, i];
                Color New = _newPropagation[0][0][j, i];
                if (old.R != New.R || old.G != New.G || old.B != New.B)
                    Console.WriteLine($"Forward Old: {old} \t New: {New}");
            }
        }
    }

    public void Backward(Vector gradient, float testLearningRate, float propLearningRate)
    {
        FeatureMap[][] dL_dP = new FeatureMap[1][];
        dL_dP[0] = new FeatureMap[1];
        dL_dP[0][0] = new FeatureMap(_oldPropagation[0][0].Width, _oldPropagation[0][0].Length, new Color(0.5f, -0.5f, 1f));
        _oldPropagation = _layer.Backwards(_initialInput, dL_dP, 1);
        _newPropagation = _layerGPU.Backwards(_initialInput, dL_dP, 1);

        for (int i = 0; i < _oldPropagation[0][0].Width; i++)
        {
            for (int j = 0; j < _oldPropagation[0][0].Length; j++)
            {
                Color old = _oldPropagation[0][0][j, i];
                Color New = _newPropagation[0][0][j, i];
                if (old.R != New.R || old.G != New.G || old.B != New.B)
                    Console.WriteLine($"Backwards Old: {old} \t New: {New}");
            }
        }
    }
}

