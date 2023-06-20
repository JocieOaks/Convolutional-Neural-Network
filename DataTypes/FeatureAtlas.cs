namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="FeatureAtlas"/> class is a class for collecting and organizing multiple <see cref="FeatureMap"/>s.
    /// </summary>
    public class FeatureAtlas
    {
        public int Dimensions { get; }
        public int BatchSize { get; }

        private readonly FeatureMap[,] _featureMaps;
    }
}