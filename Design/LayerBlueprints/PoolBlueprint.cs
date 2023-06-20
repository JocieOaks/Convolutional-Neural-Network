using ConvolutionalNeuralNetwork.Layers;

namespace ConvolutionalNeuralNetwork.Design.LayerBlueprints
{
    public readonly struct PoolBlueprint : ILayerBlueprint
    {
        public IPrimaryLayer Create()
        {
            return new AveragePool(FilterSize);
        }

        public int? InputDimensions(int outputDimensions)
        {
            return outputDimensions;
        }

        public (int, int) InputResolution((int width, int length) output)
        {
            return (output.width * FilterSize, output.length * FilterSize);
        }

        public int? OutputDimensions(int inputDimensions)
        {
            return inputDimensions;
        }

        public (int, int) OutputResolution((int width, int length) input)
        {
            return (input.width / FilterSize, input.length / FilterSize);
        }

        public int FilterSize { get; init; }
    }
}