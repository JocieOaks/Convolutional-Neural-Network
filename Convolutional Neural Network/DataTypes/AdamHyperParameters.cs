using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    /// <summary>
    /// The <see cref="AdamHyperParameters"/> class contains values for updating the <see cref="Weights"/> of a <see cref="Network"/>.
    /// </summary>
    public class AdamHyperParameters
    {
        private float _correctedLearningRate;
        [JsonProperty] private float _learningRate = 0.0001f;
        [JsonProperty] private int _updates;

        /// <value>Determines how strongly previous results affect the first moment of <see cref="Weights"/>.</value>
        public float FirstMomentDecay { get; init; } = 0.9f;

        /// <value>The learning rate of a <see cref="Network"/>, determining how quickly <see cref="Weights"/> are updated.</value>
        [JsonIgnore] public float LearningRate { get => _correctedLearningRate; init => _learningRate = value; }

        /// <value>Determines how strongly previous results affect the second moment of <see cref="Weights"/>.</value>
        public float SecondMomentDecay { get; init; } = 0.999f;

        /// <summary>
        /// Calculates the learning rate with the correction for moment bias.
        /// </summary>
        public void Update()
        {
            _updates++;
            _correctedLearningRate = _learningRate * MathF.Sqrt(1 - MathF.Pow(SecondMomentDecay, _updates)) / (1 - MathF.Pow(FirstMomentDecay, _updates));
        }
    }
}
