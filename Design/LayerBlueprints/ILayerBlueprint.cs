using ConvolutionalNeuralNetwork.Layers;

namespace ConvolutionalNeuralNetwork.Design.LayerBlueprints
{
    public interface ILayerBlueprint
    {
        (int, int) OutputResolution((int width, int length) input);

        (int, int) InputResolution((int width, int length) output);

        int? OutputDimensions(int inputDimensions);

        int? InputDimensions(int outputDimensions);

        IPrimaryLayer Create();
    }
}