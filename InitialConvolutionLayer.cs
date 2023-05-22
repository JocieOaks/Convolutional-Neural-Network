// See https://aka.ms/new-console-template for more information

[Serializable]
public class InitialConvolutionLayer : ConvolutionalLayer
{
    public InitialConvolutionLayer(int dimensions, int kernalSize, int stride, ref FeatureMap[][] input) : base(dimensions, kernalSize, stride, ref input)
    {
    }

    public FeatureMap[][] Forward(FeatureMap[] input)
    {
        for (int i = 0; i < _dimensions; i++)
        {
            Forward(input, _convoluted[i], _kernals[i]);
        }

        while (_threadsWorking > 0)
            Thread.Sleep(100);

        return _convoluted;
    }

    public FeatureMap[][] Backwards(FeatureMap[] input, FeatureMap[][] dL_dP, float learningRate)
    { 
        for (int i = 0; i < _dimensions; i++)
        {
            Backwards(input, _kernals[i], _dL_dK[i], dL_dP[i], _dL_dPNext[i], learningRate);
        }
        while (_threadsWorking > 0)
            Thread.Sleep(100);

        return _dL_dPNext;
    }
}
