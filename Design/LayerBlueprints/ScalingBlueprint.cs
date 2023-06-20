using ConvolutionalNeuralNetwork.Layers;

namespace ConvolutionalNeuralNetwork.Design.LayerBlueprints
{
    public readonly struct ScalingBlueprint : ILayerBlueprint
    {
        public IPrimaryLayer Create()
        {
            Scaling scalingLayer = new();
            scalingLayer.SetScale(ScalingMultiplier, ScalingMultiplier);
            return scalingLayer;
        }

        public int? InputDimensions(int outputDimensions)
        {
            throw new NotImplementedException();
        }

        public (int, int) InputResolution((int width, int length) output)
        {
            throw new NotImplementedException();
        }

        public int? OutputDimensions(int inputDimensions)
        {
            throw new NotImplementedException();
        }

        public (int, int) OutputResolution((int width, int length) input)
        {
            throw new NotImplementedException();
        }

        public int ScalingMultiplier { get; init; }
    }
}