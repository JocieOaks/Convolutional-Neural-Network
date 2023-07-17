using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ConvolutionalNeuralNetwork.DataTypes;

namespace ConvolutionalNeuralNetwork.Layers
{
    public static class HyperTan
    {
        public static FeatureMap[,] Forward(FeatureMap[,] inputs)
        {
            FeatureMap[,] output = new FeatureMap[inputs.GetLength(0), inputs.GetLength(1)];
            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                for (int j = 0; j < inputs.GetLength(1); j++)
                {
                    output[i, j] = new FeatureMap(inputs[i, j].Width, inputs[i, j].Length);
                    for (int y = 0; y < inputs[i, j].Length; y++)
                    {
                        for (int x = 0; x < inputs[i, j].Width; x++)
                        {
                            output[i, j][x, y] = MathF.Tanh(-inputs[i, j][x, y]);
                        }
                    }
                }
            }

            return output;
        }

        public static FeatureMap[,] Backward(FeatureMap[,] inputs, FeatureMap[,] inGradients)
        {
            FeatureMap[,] outGradients = new FeatureMap[inputs.GetLength(0), inputs.GetLength(1)];
            for (int i = 0; i < outGradients.GetLength(0); i++)
            {
                for (int j = 0; j < inputs.GetLength(1); j++)
                {
                    outGradients[i, j] = new(inGradients[i, j].Width, inGradients[i, j].Length);
                    for (int y = 0; y < inGradients[i, j].Length; y++)
                    {
                        for (int x = 0; x < inGradients[i, j].Width; x++)
                        {
                            float cosh = MathF.Cosh(-inputs[i, j][x, y]);
                            float dTanh = MathF.Pow(cosh, -2);
                            outGradients[i, j][x, y] = dTanh * inGradients[i, j][x, y];
                        }
                    }
                }
            }
            return outGradients;
        }
    }
}
