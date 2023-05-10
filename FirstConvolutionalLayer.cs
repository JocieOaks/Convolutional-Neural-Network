// See https://aka.ms/new-console-template for more information
public class FirstConvolutionalLayer<T> : ConvolutionalLayer<T> where T : IDot<T>, new()
{
    public FirstConvolutionalLayer(int kernalsNum, int kernalSize, int stride) : base(kernalsNum, kernalSize, stride)
    {
    }

    public T[][,] Forward(T[,] input)
    {
        T[][,] convoluted = new T[_kernalNum][,];
        for (int kernalIndex = 0; kernalIndex < _kernalNum; kernalIndex++)
        {
            convoluted[kernalIndex] = Forward(input, kernalIndex);
        }

        return convoluted;
    }
}
