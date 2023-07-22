using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="ILayer"/> interface is the extracted interface for <see cref="Layer"/>.
    /// </summary>
    public interface ILayer
    {
        /// <value>The name of the <see cref="Layer"/>, used for logging.</value>
        string Name { get; }

        /// <summary>
        /// Backpropagates through the <see cref="Layer"/> updating any layer weights, and calculating the outgoing gradient that is
        /// shared with the previous layer.
        /// </summary>
        /// <param name="learningRate">The overall learning rate for the layer updates, corrected for the influence of bias in the first and second moments.</param>
        /// <param name="firstMomentDecay">The exponential decay rate for the first moment.</param>
        /// <param name="secondMomentDecay">The exponential decay rate for the second moment.</param>
        void Backwards(float learningRate, float firstMomentDecay, float secondMomentDecay);

        /// <summary>
        /// Forward propagates through the <see cref="Layer"/> calculating the output <see cref="FeatureMap"/> that is shared with
        /// the next layer.
        /// </summary>
        void Forward();

        /// <summary>
        /// Initializes the <see cref="Layer"/> for the data set being used.
        /// </summary>
        /// <param name="inputs">The previous <see cref="Layer"/>'s output.</param>
        /// <param name="outGradients">The previous <see cref="Layer"/>'s inGradient.</param>
        /// <returns>Returns the output and inGradient to share with the next <see cref="Layer"/>.</returns>
        Shape Startup(Shape inputShapes, IOBuffers buffers, int batchSize);

        /// <summary>
        /// Reset's the current <see cref="Layer"/> to it's initial weights or initial random weights.
        /// </summary>
        void Reset();
    }

    /// <summary>
    /// The <see cref="IPrimaryLayer"/> interface is for <see cref="Layer"/>s that perform significant manipulation to the <see cref="FeatureMap"/>s,
    /// and are thus fundamental to the architecture of a <see cref="Network"/>.
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

    /// <summary>
    /// The <see cref="IUnchangedLayer"/> interface is for <see cref="Layer"/>s where the direct input is the same as the direct output. The layer may perform some alternate
    /// task, such as modifying the shape of the output (but the single dimensional values remain constant) or copying the input for a later layer.
    /// </summary>
    public interface IUnchangedLayer : ILayer
    { }
}