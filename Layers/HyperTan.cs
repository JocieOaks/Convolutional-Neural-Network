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
        public static void Forward(FeatureMap[] inputs)
        {
            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                for (int y = 0; y < inputs[i].Length; y++)
                {
                    for (int x = 0; x < inputs[i].Width; x++)
                    {
                        inputs[i][x, y] = MathF.Tanh(inputs[i][x, y]);
                    }
                }
            }
        }

        public static void Backward(FeatureMap[] outputs, FeatureMap[] inGradients)
        {
            for (int i = 0; i < outputs.Length; i++)
            {
                for (int y = 0; y < inGradients[i].Length; y++)
                {
                    for (int x = 0; x < inGradients[i].Width; x++)
                    {
                        inGradients[i][x, y] = inGradients[i][x, y] * (1 - MathF.Pow(outputs[i][x, y], 2));
                    }
                }
            }
        }
    }
}
