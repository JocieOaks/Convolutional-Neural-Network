using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

public static class VectorNormalizationLayer
{

    public static Vector[] Forward(Vector[] input)
    {
        Vector[] vectors = new Vector[input.Length];
        for(int i = 0; i < vectors.Length; i++)
        {
            vectors[i] = Forward(input[i]);
        }
        return vectors;
    }

    public static Vector Forward(Vector input)
    {
        return input.Normalized();
    }

    public static Vector[] Backwards(Vector[] input, Vector[] inGradients)
    {
        Vector[] outGradients = new Vector[input.Length];
        for(int i = 0; i < outGradients.Length ; i++)
        {
            outGradients[i] = Backwards(input[i], inGradients[i]);
        }
        return outGradients;
    }

    public static Vector Backwards(Vector input, Vector inGradient)
    {
        float magnitude = input.Magnitude;
        float invMagnitude = 1 / magnitude;

        Vector outGradient = new Vector(inGradient.Length);

        for(int i = 0; i < inGradient.Length; i++)
        {
            outGradient[i] = (magnitude - input[i] * input[i] * invMagnitude) * invMagnitude * invMagnitude * inGradient[i];
        }

        return outGradient;
    }

}

