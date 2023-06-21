using ConvolutionalNeuralNetwork.Layers;

namespace ConvolutionalNeuralNetwork.Design
{
    /// <summary>
    /// Represent various the <see cref="Layer"/> used for normalization or activation after a convolution for setting up how
    /// the layers should be inserted into a <see cref="Network"/>.
    /// </summary>
    public enum NormalizationLayers
    {
        Activation,
        BatchNormalization,
        Dropout
    }

    /// <summary>
    /// The <see cref="ActivationPattern"/> struct details a pattern for what <see cref="ISecondaryLayer"/> should follow after an
    /// <see cref="IPrimaryLayer"/>.
    /// </summary>
    public struct ActivationPattern
    {
        private NormalizationLayers[] _pattern;
        private float _dropoutRate;

        /// <summary>
        /// Initializes a new instance of the <see cref="ActivationPattern"/> class.
        /// </summary>
        /// <param name="pattern">An array detailing the pattern of <see cref="ISecondaryLayer"/>s.</param>
        /// <param name="dropoutRate">The dropout rate for any <see cref="Dropout"/> layers used.</param>
        public ActivationPattern(NormalizationLayers[] pattern, float dropoutRate)
        {
            _pattern = pattern;
            _dropoutRate = dropoutRate;
        }

        /// <summary>
        /// Iteratively creates new <see cref="ISecondaryLayer"/>s following the defined pattern.
        /// </summary>
        /// <returns>Yield returns a new <see cref="ISecondaryLayer"/>.</returns>
        public IEnumerable<ISecondaryLayer> GetLayers()
        {
            foreach (var layer in _pattern)
            {
                yield return layer switch
                {
                    NormalizationLayers.Activation => new ReLUActivation(),
                    NormalizationLayers.BatchNormalization => new BatchNormalization(),
                    NormalizationLayers.Dropout => new Dropout(_dropoutRate)
                };
            }
        }
    }
}