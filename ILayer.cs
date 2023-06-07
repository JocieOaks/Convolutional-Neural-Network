public interface ILayer
{
    string Name { get; }

    FeatureMap[,] Backwards(FeatureMap[,] input, FeatureMap[,] inGradient, float learningRate);

    FeatureMap[,] Forward(FeatureMap[,] input);

    FeatureMap[,] Startup(FeatureMap[,] input);

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