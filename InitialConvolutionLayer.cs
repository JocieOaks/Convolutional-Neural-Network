// See https://aka.ms/new-console-template for more information

[Serializable]
public class InitialConvolutionLayer<T> : ConvolutionalLayer<T> where T : IDot<T>, new()
{
    public InitialConvolutionLayer(int kernalsNum, int kernalSize, int stride) : base(kernalsNum, kernalSize, stride)
    {
    }

    public T[][,] Forward(T[,] input)
    {
        T[][,] convoluted = new T[_kernalNum][,];
        for (int i = 0; i < _kernalNum; i++)
        {
            convoluted[i] = Forward(input, _kernals[i]);
        }

        return convoluted;
    }

    public T[][,] Backwards(T[][,] error, T[,] input, float alpha)
    {
        T[][,] corrections = new T[_kernalNum][,];

        for (int i = 0; i < _kernalNum; i++)
        {
            corrections[i] = Backwards(error[i], input, _kernals[i], alpha);
        }

        return corrections;
    }
}
