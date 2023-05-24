using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

public static class VectorNormalizationLayer
{

    public static Vector Forward(Vector input)
    {
        return input.Normalized();
    }

    public static Vector Backwards(Vector input, Vector dL_dV)
    {
        float magnitude = input.Magnitude;
        float invMagnitude = 1 / magnitude;

        Vector dL_dVNext = new Vector(dL_dV.Length);

        for(int i = 0; i < dL_dV.Length; i++)
        {
            dL_dVNext[i] = (magnitude - input[i] * input[i] * invMagnitude) * invMagnitude * invMagnitude * dL_dV[i];
        }

        return dL_dVNext;
    }

}

