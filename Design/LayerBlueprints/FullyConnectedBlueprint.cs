using ConvolutionalNeuralNetwork.Layers;

namespace ConvolutionalNeuralNetwork.Design.LayerBlueprints
{
    public readonly struct FullyConnectedBlueprint : ILayerBlueprint
    {
        public IPrimaryLayer Create()
        {
            FullyConnected layer = new();
            if (DimensionMultiplier.HasValue)
                layer.SetOutputMultiplier(DimensionMultiplier.Value);
            return layer;
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
            return (output.width, output.length);
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
            return (input.width, input.length);
        }

        public int? DimensionMultiplier { get; init; }
    }
}