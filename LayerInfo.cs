// See https://aka.ms/new-console-template for more information
#nullable disable

public readonly struct LayerInfo
{
    public LayerInfo() { }

    public int InputWidth { get; init; }
    public int InputLength { get; init; }
    public int InputArea => InputWidth * InputLength;
    public float InverseKSquared { get; init; }
    public int KernalSize { get; init; }
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

    public int KernalIndex(int x, int y)
    {
        return y * KernalSize + x;
    }
}