using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;


public class BackPropogationTest
{

    readonly FeatureMap[][] _initialInput;

    FeatureMap[][] oldPropogation;
    FeatureMap[][] newPropogation;

    ConvolutionalLayer _testLayer;
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
                    _initialInput[0][0][j, i] = new Color(CLIP.Random.Next(-1,2), CLIP.Random.Next(-1, 2), CLIP.Random.Next(-1, 2));
                }
            }
        }
        FeatureMap[][] current = _initialInput;
        _testLayer = new ConvolutionalLayer(3, 1, ref current);
        //_impactLayer = new ConvolutionalLayer(1, 3, 1);
    }

    public float Test(float testLearningRate, float propLearningRate)
    {
        Forward();
        //float loss = Loss();
        //Vector dL_dP = Gradient(loss);
        Backward(new Vector(4), testLearningRate, propLearningRate);
        return 0;
    }

    public float Loss()
    {
        float sum = 0;
        //for(int i = 0; i < _finalOutput.Length; i++)
        {
        //    sum += MathF.Pow(_finalOutput[i], 2);
        }

        return sum;
    }

    public Vector Gradient(float loss)
    {
        Vector dL_dP = new Vector(4) * 2;
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
    }

    public void Backward(Vector gradient, float testLearningRate, float propLearningRate)
    {
        //dL_dP[0] = _testLayer.Backwards(_intermediate[0], gradient, (float)testLearningRate);
    }
}

