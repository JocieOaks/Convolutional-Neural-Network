using ConvolutionalNeuralNetwork.Layers;

namespace ConvolutionalNeuralNetwork.Design.LayerBlueprints
{
    public readonly struct ConvolutionBlueprint : ILayerBlueprint
    {
        public IPrimaryLayer Create()
        {
            if (Key)
                return new LatentConvolution(FilterSize, Stride, DimensionMultiplier.HasValue ? DimensionMultiplier.Value : 1);
            return new Convolution(FilterSize, Stride, DimensionMultiplier.HasValue ? DimensionMultiplier.Value : 1);
        }

        public int? InputDimensions(int outputDimensions)
        {
            if (!DimensionMultiplier.HasValue)
                return null;
            if (DimensionMultiplier > 0)
            {
                return outputDimensions % DimensionMultiplier != 0 ? null : outputDimensions / DimensionMultiplier;
            }
            else
            {
                return outputDimensions * -DimensionMultiplier;
            }
        }

        public (int, int) InputResolution((int width, int length) output)
        {
            int inputWidth = Stride * (output.width - 2) + 1 + FilterSize;
            int inputLength = Stride * (output.length - 2) + 1 + FilterSize;

            return (inputWidth, inputLength);
        }

        public int? OutputDimensions(int inputDimensions)
        {
            if (!DimensionMultiplier.HasValue)
                return null;
            if (DimensionMultiplier > 0)
            {
                return inputDimensions * DimensionMultiplier;
            }
            else
            {
                return inputDimensions % -DimensionMultiplier != 0 ? null : inputDimensions / -DimensionMultiplier;
            }
        }

        public (int, int) OutputResolution((int width, int length) input)
        {
            int outputWidth = 2 + (input.width - FilterSize - 1) / Stride;
            int outputLength = 2 + (input.length - FilterSize - 1) / Stride;
            return (outputWidth, outputLength);
        }

        public int FilterSize { get; init; }
        public int Stride { get; init; }
        public int? DimensionMultiplier { get; init; }
        public bool Key { get; init; }
    }
}