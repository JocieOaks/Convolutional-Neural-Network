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
        public FeatureMap[] Image { get; init; }

        /// <value>The name of the image if it has one. Used to match label and image from separate files.</value>
        public string ImageName { get; init; }

        public Vector LabelVector(int latentDimensions = 0)
        {
            Vector vector = new(Bools.Length + Floats.Length + latentDimensions);
            for (int i = 0; i < Bools.Length; i++)
            {
                vector[i] = Bools[i] ? 1 : -1;
            }
            for (int i = 0; i < Floats.Length; i++)
            {
                vector[Bools.Length + i] = Floats[i] * 2 - 1;
            }
            int labels = Bools.Length + Floats.Length;
            for(int i = 0; i < latentDimensions; i++)
            {
                vector[labels + i] = Utility.RandomGauss(0, 1);
            }
            return vector;
        }
    }
}