using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


public static class Augmentations
{
    public static FeatureMap GaussianNoise(FeatureMap featureMap)
    {
        FeatureMap newFeatureMap = new FeatureMap(featureMap.Width, featureMap.Length);

        for (int j = 0; j < featureMap.Length; j++)
        {
            for (int i = 0; i < featureMap.Width; i++)
            {
                newFeatureMap[i, j] = featureMap[i, j] + Color.RandomGauss(0, 0.1f);
            }
        }

        return newFeatureMap;
    }

    public static FeatureMap RandomSaturation(FeatureMap featureMap)
    {
        float saturation = ConvolutionalNeuralNetwork.RandomGauss(0, 0.1f);

        FeatureMap newFeatureMap = new FeatureMap(featureMap.Width, featureMap.Length);

        for (int j = 0; j < featureMap.Length; j++)
        {
            for (int i = 0; i < featureMap.Width; i++)
            {
                Color pixel = featureMap[i, j];
                float L = 0.3f * pixel.R + 0.6f * pixel.G + 0.1f * pixel.B;

                float R = pixel.R + saturation * (L - pixel.R);
                float G = pixel.G + saturation * (L - pixel.G);
                float B = pixel.B + saturation * (L - pixel.B);

                newFeatureMap[i, j] = new Color(R, G, B);
            }
        }

        return newFeatureMap;
    }

    public static FeatureMap RandomBrightness(FeatureMap featureMap)
    {
        float brightness = ConvolutionalNeuralNetwork.RandomGauss(1, 0.1f);

        FeatureMap newFeatureMap = new FeatureMap(featureMap.Width, featureMap.Length);

        for (int j = 0; j < featureMap.Length; j++)
        {
            for (int i = 0; i < featureMap.Width; i++)
            {
                newFeatureMap[i, j] = featureMap[i, j] * brightness;
            }
        }

        return newFeatureMap;
    }

    public static FeatureMap HorizontalFlip(FeatureMap featureMap)
    {
        FeatureMap newFeatureMap = new FeatureMap(featureMap.Width, featureMap.Length);

        for (int j = 0; j < featureMap.Length; j++)
        {
            for (int i = 0; i < featureMap.Width; i++)
            {
                newFeatureMap[i, j] = featureMap[featureMap.Width - i - 1, j];
            }
        }

        return newFeatureMap;
    }
}
