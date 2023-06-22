namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="ImageInput"/> struct contains an image as a <see cref="FeatureMap"/> and its labels.
    /// </summary>
    public readonly struct ImageInput
    {
        /// <value>An array of bools constituing the image labels that are discretely true or false.</value>
        public bool[] Bools { get; init; }

        /// <value>An array of floats constituting the image labels that have a continuous range of values.</value>
        public float[] Floats { get; init; }

        /// <value>The image a <see cref="FeatureMap"/>.</value>
        public FeatureMap Image { get; init; }

        /// <value>The name of the image if it has one. Used to match label and image from separate files.</value>
        public string ImageName { get; init; }
    }
}