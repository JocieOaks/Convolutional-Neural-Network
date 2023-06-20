namespace ConvolutionalNeuralNetwork.DataTypes
{
    public readonly struct ImageInput
    {
        public bool[] Bools { get; init; }
        public float[] Floats { get; init; }
        public FeatureMap Image { get; init; }
        public string ImageName { get; init; }

        public ImageInput(ImageInput input)
        {
            Bools = input.Bools;
            Floats = input.Floats;
            Image = input.Image;
            ImageName = input.ImageName;
        }
    }
}