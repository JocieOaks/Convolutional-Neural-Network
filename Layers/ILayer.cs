using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers
{
    public interface ILayer
    {
        string Name { get; }

        public FeatureMap[,] Outputs { get; }

        public int OutputDimensions { get; }

        void Backwards(float learningRate);

        void Forward();

        (FeatureMap[,], FeatureMap[,]) Startup(FeatureMap[,] input, FeatureMap[,] outGradients);

        void Reset();
    }

    /// <summary>
    /// The <see cref="IPrimaryLayer"/> interface is for <see cref="Layer"/>s that perform significant manipulation to the <see cref="FeatureMap"/>s.
    /// </summary>
    public interface IPrimaryLayer : ILayer
    { }

    /// <summary>
    /// The <see cref="IStructuralLayer"/> interface is for <see cref="Layer"/>s that change the structure of a <see cref="FeatureMap"/>, such as by scaling the map,
    /// but do not manipulate the actual features, and thus does not need to be activated.
    /// </summary>
    public interface IStructuralLayer : IPrimaryLayer
    { }

    /// <summary>
    /// The <see cref="ISecondaryLayer"/> interface is for <see cref="Layer"/>s that normalize or activate the <see cref="FeatureMap"/>s.
    /// These typically always follow <see cref="IPrimaryLayer"/>s.
    /// </summary>
    public interface ISecondaryLayer : ILayer
    { }
}