public interface ILayerInfo
{
    int InputWidth { get; }
    int InputLength { get; }
    int InputArea => InputWidth * InputLength;
    float InverseKSquared { get; }
    int FilterSize { get; }
    int OutputWidth { get; }
    int OutputLength { get; }
    public int OutputArea => OutputWidth * OutputLength;
    int Stride { get; }
}

public readonly struct LayerInfo : ILayerInfo
{
    public LayerInfo()
    { }

    public int InputWidth { get; init; }
    public int InputLength { get; init; }
    public int InputArea => InputWidth * InputLength;
    public float InverseKSquared { get; init; }
    public int FilterSize { get; init; }
    public int OutputWidth { get; init; }
    public int OutputLength { get; init; }
    public int OutputArea => OutputWidth * OutputLength;
    public int Stride { get; init; }

    public int OutputIndex(int x, int y)
    {
        return y * OutputWidth + x;
    }

    public bool TryGetInputIndex(int strideX, int x, int strideY, int y, out int index)
    {
        x += strideX * Stride;
        y += strideY * Stride;
        index = y * InputWidth + x;
        return x < InputWidth && y < InputLength;
    }

    public int FilterIndex(int x, int y)
    {
        return y * FilterSize + x;
    }
}

public readonly struct SingleLayerInfo : ILayerInfo
{
    public int Width { get; init; }
    public int Length { get; init; }
    public int Area => Width * Length;

    public int InputWidth => Width;

    public int InputLength => Length;

    public float InverseKSquared => 1;

    public int FilterSize => 1;

    public int OutputWidth => Width;

    public int OutputLength => Length;

    public int Stride => 1;

    public int Index(int x, int y)
    {
        return y * Width + x;
    }
}