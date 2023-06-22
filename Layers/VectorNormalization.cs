using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers
{
    /// <summary>
    /// The <see cref="VectorNormalization"/> class is a static class that normalizes a <see cref="Vector"/>
    /// and calculates the gradients for the original <see cref="Vector"/> from the nomralized <see cref="Vector"/> gradients.
    /// </summary>
    public static class VectorNormalization
    {
        /// <summary>
        /// Normalizes the input <see cref="Vector"/>s.
        /// </summary>
        /// <param name="input">The <see cref="Vector"/>s being normalized.</param>
        /// <returns>Return the normalized <see cref="Vector"/>s.</returns>
        public static Vector[] Forward(Vector[] input)
        {
            Vector[] vectors = new Vector[input.Length];
            for (int i = 0; i < vectors.Length; i++)
            {
                vectors[i] = input[i].Normalized();
            }
            return vectors;
        }

        /// <summary>
        /// Calculates the gradients of the original <see cref="Vector"/>s from the gradients of the normalized <see cref="Vector"/>s.
        /// </summary>
        /// <param name="inputs">The normalized <see cref="Vector"/>s.</param>
        /// <param name="inGradients">The normalized <see cref="Vector"/>s' gradients.</param>
        /// <returns>Returns the gradients of the original <see cref="Vector"/>s.</returns>
        public static Vector[] Backwards(Vector[] inputs, Vector[] inGradients)
        {
            Vector[] outGradients = new Vector[inputs.Length];
            for (int i = 0; i < outGradients.Length; i++)
            {
                outGradients[i] = new(inGradients[i].Length);
                float magnitude = inputs[i].Magnitude;
                float invMagnitude = 1 / magnitude;

                for (int j = 0; j < inGradients[i].Length; j++)
                {
                    outGradients[i][j] = (magnitude - inputs[i][j] * inputs[i][j] * invMagnitude) * invMagnitude * invMagnitude * inGradients[i][j];
                }
            }
            return outGradients;
        }
    }
}