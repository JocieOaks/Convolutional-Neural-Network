using ConvolutionalNeuralNetwork.DataTypes;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers
{
    public static class Sigmoid
    {
        public static Vector[] Forward(Vector[] inputs)
        {
            Vector[] output = new Vector[inputs.Length];
            for(int i = 0; i < inputs.Length; i++)
            {
                output[i] = new Vector(inputs[i].Length);
                for (int j = 0; j < inputs[i].Length; j++)
                {
                    output[i][j] = 1 / (1 + MathF.Exp(-inputs[i][j]));
                }
            }

            return output;
        }

        public static Vector[] Backward(Vector[] inputs, Vector[] outputs, Vector[] inGradients)
        {
            Vector[] outGradients = new Vector[inputs.Length];
            for (int i = 0; i < outGradients.Length; i++)
            {
                outGradients[i] = new(inGradients[i].Length);
                for (int j = 0; j < inGradients[i].Length; j++)
                {
                    float exp = MathF.Exp(inputs[i][j]);
                    outGradients[i][j] = exp * outputs[i][j] * outputs[i][j] * inGradients[i][j];
                }
            }
            return outGradients;
        }
    }
}
