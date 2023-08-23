using Newtonsoft.Json;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    public class AdamHyperParameters
    {
        [JsonProperty] private float _learningRate = 0.0001f;
        public float FirstMomentDecay { get; init; } = 0.9f;
        public float SecondMomentDecay { get; init; } = 0.999f;
        [JsonIgnore] public float LearningRate { get => _correctedLearningRate; init => _learningRate = value; }
        private float _correctedLearningRate;
        [JsonProperty] private int _updates = 0;

        [JsonProperty] private readonly float _learningRateDecay = 0.464f;

        /// <summary>
        /// Calculates the learning rate with the correction for moment bias.
        /// </summary>
        /// <param name="learningRate">The overall learning rate for the layer updates, corrected for the influence of bias in the first and second moments.</param>
        /// 
        /// 
        /// <returns>Returns the learning rate multiplied by the correction term.</returns>
        public void Update(bool updateWeights = true)
        {
            _updates++;
            _correctedLearningRate = _learningRate * MathF.Pow(_learningRateDecay, _updates / 75e4f) * MathF.Sqrt(1 - MathF.Pow(SecondMomentDecay, _updates)) / (1 - MathF.Pow(FirstMomentDecay, _updates));
        }

        public AdamHyperParameters Copy()
        {
            return new AdamHyperParameters()
            {
                LearningRate = _learningRate,
                FirstMomentDecay = FirstMomentDecay,
                SecondMomentDecay = SecondMomentDecay
            };
        }
    }
}
