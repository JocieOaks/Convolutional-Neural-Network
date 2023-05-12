// See https://aka.ms/new-console-template for more information

[Serializable]
public class InitialConvolutionLayer : ConvolutionalLayer
{
    public InitialConvolutionLayer(int kernalsNum, int kernalSize, int stride) : base(kernalsNum, kernalSize, stride)
    {
    }

    public FeatureMap[][] Forward(FeatureMap[] input)
    {
        FeatureMap[][] convoluted = new FeatureMap[_dimensions][];
        for (int i = 0; i < _dimensions; i++)
        {
            convoluted[i] = Forward(input, _kernals[i]);
        }

        return convoluted;
    }

    public FeatureMap[][] Backwards(FeatureMap[][] error, FeatureMap[] input, float alpha)
    {
        FeatureMap[][] corrections = new FeatureMap[_dimensions][];

        for (int i = 0; i < _dimensions; i++)
        {
            corrections[i] = Backwards(error[i], input, _kernals[i], alpha);
        }

        return corrections;
    }
}
