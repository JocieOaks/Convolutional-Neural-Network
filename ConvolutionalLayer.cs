// See https://aka.ms/new-console-template for more information
#nullable disable

using Newtonsoft.Json;
using System.Runtime.Serialization;

[Serializable]
public class ConvolutionalLayer : Layer
{
    [JsonProperty]
    protected Color[][,] _kernals;
    protected Color[][,] _dL_dK;

    float _invK2;
    protected FeatureMap[][] _convoluted;
    protected FeatureMap[][] _dL_dPNext;

    protected const int CLAMP = 1;
    protected const float LEARNINGMULTIPLIER = 1f;

    protected int _threadsWorking;

    public ConvolutionalLayer(int kernalSize, int stride, ref FeatureMap[][] input) : base(input.Length, kernalSize, stride)
    {
        _kernals = new Color[_dimensions][,];
        _dL_dK = new Color[_dimensions][,];

        float variance = 0.333f / (_dimensions * kernalSize * kernalSize);
        float stdDev = MathF.Sqrt(variance);

        for (int i = 0; i < _dimensions; i++)
        {
            _kernals[i] = new Color[_kernalSize, _kernalSize];
            _dL_dK[i] = new Color[kernalSize, _kernalSize];

            for (int j = 0; j < kernalSize; j++)
            {
                for (int k = 0; k < kernalSize; k++)
                {
                    _kernals[i][j, k] = Color.RandomGauss(0, stdDev);
                }
            }
        }

        _convoluted = new FeatureMap[_dimensions][];
        _dL_dPNext = new FeatureMap[_dimensions][];
        for (int i = 0; i < _dimensions; i++)
        {
            Pad(input[i]);
            _convoluted[i] = new FeatureMap[input[i].Length];
            _dL_dPNext[i] = new FeatureMap[input[i].Length];
            for (int j = 0; j < input[i].Length; j++)
            {
                _dL_dPNext[i][j] = new FeatureMap(input[i][j].Width, input[i][j].Length);
                int width = (input[i][j].Width - kernalSize) / stride;
                int length = (input[i][j].Length - kernalSize) / stride;
                _convoluted[i][j] = new FeatureMap(width, length);
            }
        }
        input = _convoluted;

        _invK2 = 1f / (kernalSize * kernalSize);
    }

    public ConvolutionalLayer() : base(0, 0, 0) { }

    [OnDeserialized]
    public void OnDeserialized(StreamingContext context)
    {
        _invK2 = 1f / (_kernalSize * _kernalSize);
    }

    public override FeatureMap[][] Backwards(FeatureMap[][] input, FeatureMap[][] dL_dP, float learningRate)
    {
        for (int i = 0; i < _dimensions; i++)
        {
            Backwards(input[i], _kernals[i], _dL_dK[i], dL_dP[i], _dL_dPNext[i]);
        }

        do
            Thread.Sleep(100);
        while (_threadsWorking > 0);

        for (int i = 0; i < _dimensions; i++)
        {
            for (int j = 0; j < _kernalSize; j++)
            {
                for (int k = 0; k < _kernalSize; k++)
                {
                    _kernals[i][j, k] -= learningRate * LEARNINGMULTIPLIER * _dL_dK[i][j, k];
                }
            }
        }

        return _dL_dPNext;
    }

    public override FeatureMap[][] Forward(FeatureMap[][] input)
    {
        for (int i = 0; i < _dimensions; i++)
        {
            Forward(input[i], _convoluted[i], _kernals[i]);
        }

        do
            Thread.Sleep(100);
        while (_threadsWorking > 0);

        return _convoluted;
    }

    protected void Backwards(FeatureMap[] input, Color[,] kernal, Color[,] dL_dK, FeatureMap[] dL_dP, FeatureMap[] dL_dPNext)
    {
        for(int i = 0; i < _kernalSize; i++)
        {
            for(int j = 0; j < _kernalSize; j++)
            {
                dL_dK[i, j] = new Color(0, 0, 0);
            }
        }

        int batchLength = input.Length;
        for (int i = 0; i < batchLength; i++)
        {
            ThreadPool.QueueUserWorkItem(Backwards, (input[i], kernal, dL_dK, dL_dP[i], dL_dPNext[i]));
        }
    }

    protected void Backwards(object stateInfo)
    {
        if (stateInfo == null)
            throw new ArgumentNullException(nameof(stateInfo));
        (FeatureMap input, Color[,] kernal, Color[,] dL_dK, FeatureMap dL_dP, FeatureMap dL_dPNext) = ((FeatureMap, Color[,], Color[,], FeatureMap, FeatureMap))stateInfo;
        Interlocked.Increment(ref _threadsWorking);
        lock (dL_dPNext)
        {
            lock (dL_dK)
            {
                for (int strideX = 0; strideX < dL_dP.Width; strideX++)
                {
                    for (int strideY = 0; strideY < dL_dP.Length; strideY++)
                    {
                        for (int kernalX = 0; kernalX < _kernalSize; kernalX++)
                        {
                            for (int kernalY = 0; kernalY < _kernalSize; kernalY++)
                            {
                                int x = strideX * _stride + kernalX;
                                int y = strideY * _stride + kernalY;
                                Color dK = dL_dP[strideX, strideY] * input[x, y] * _invK2;
                                Color dP = dL_dP[strideX, strideX] * kernal[kernalX, kernalY] * _invK2;
                                dL_dK[kernalX, kernalY] += dK;
                                dL_dPNext[x, y] += dP;
                            }
                        }
                    }
                }
                for (int i = 0; i < _kernalSize; i++)
                {
                    for (int j = 0; j < _kernalSize; j++)
                    {
                        dL_dK[i, j] = dL_dK[i, j].Clamp(CLAMP);
                    }
                }

                for (int i = 0; i < dL_dPNext.Width; i++)
                {
                    for (int j = 0; j < dL_dPNext.Length; j++)
                    {
                        dL_dPNext[i, j] = dL_dPNext[i, j].Clamp(1f);
                    }
                }
            }
        }
        Interlocked.Decrement(ref _threadsWorking);
    }

    protected void Forward(FeatureMap[] input, FeatureMap[] convoluted, Color[,] kernal)
    {
        for (int i = 0; i < input.Length; i++)
        {
            ThreadPool.QueueUserWorkItem(Forward, (input[i], convoluted[i], kernal));
        }
    }

    protected void Forward(Object stateInfo)
    {
        if (stateInfo == null)
            throw new ArgumentNullException(nameof(stateInfo));
        (FeatureMap input, FeatureMap convoluted, Color[,] kernal) = ((FeatureMap, FeatureMap, Color[,]))stateInfo;
        Interlocked.Increment(ref _threadsWorking);
        lock (convoluted)
        {
            for (int shiftX = 0; shiftX < convoluted.Width; shiftX++)
            {
                for (int shiftY = 0; shiftY < convoluted.Length; shiftY++)
                {
                    for (int kernalX = 0; kernalX < _kernalSize; kernalX++)
                    {
                        for (int kernalY = 0; kernalY < _kernalSize; kernalY++)
                        {
                            convoluted[shiftX, shiftY] += input[shiftX * _stride + kernalX, shiftY * _stride + kernalY] * kernal[kernalX, kernalY];
                        }
                    }
                    convoluted[shiftX, shiftY] *= _invK2;
                }
            }
        }
        Interlocked.Decrement(ref _threadsWorking);
    }
}
