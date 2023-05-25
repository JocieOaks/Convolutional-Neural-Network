// See https://aka.ms/new-console-template for more information
#nullable disable

public readonly struct GPUKernalFeatures
{
    public GPUKernalFeatures(int inputWidth, int outputWidth, int kernalSize, int stride, float inverseKSquared)
    {
        InputWidth = inputWidth;
        OutputWidth = outputWidth;
        KernalSize = kernalSize;
        Stride = stride;
        InverseKSquared = inverseKSquared;
    }

    public int InputWidth { get; }
    public float InverseKSquared { get; }
    public int KernalSize { get; }
    public int OutputWidth { get; }
    public int Stride { get; }
}