using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


public static class GradientChecking
{
    public static void TestConvolutionalLayer()
    {
        FeatureMap[,] testInput = new FeatureMap[, ] { { new FeatureMap(3, 3) } };
        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                testInput[0, 0][i, j] = new Color(i, j, i - j);
            }
        }
        FeatureMap[,] testOutput = testInput;
        ConvolutionalLayer layer = new ConvolutionalLayer(3, 1, 1);
        FeatureMap[,] gradient = new FeatureMap[,] { { new FeatureMap(1, 1, new Color(1)) } };
        testOutput = layer.Forward(testInput);
        Color output = testOutput[0, 0][0, 0];
        FeatureMap[,] outGradient = layer.Backwards(testInput, gradient, 0);

        float h = 0.0001f;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Color hColor = k switch
                    {
                        0 => new Color(h, 0, 0),
                        1 => new Color(0, h, 0),
                        2 => new Color(0, 0, h)
                    };
                    testInput[0, 0][i, j] += hColor;
                    testOutput = layer.Forward(testInput);
                    Console.WriteLine($"Expected Gradient: {outGradient[0, 0][i, j][k]:f4} \t Test Gradient: {(testOutput[0, 0][0, 0][k] - output[k]) / h:f4}");
                    testInput[0, 0][i, j] -= hColor;

                }
            }
        }
    }

    public static void TestBatchNormalizationLayer()
    {
        FeatureMap[,] testInput = new FeatureMap[,] { { new FeatureMap(3, 3) } };
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                testInput[0, 0][i, j] = new Color(i, j, i - j);
            }
        }
        FeatureMap[,] testOutput = testInput;
        BatchNormalizationLayer layer = new BatchNormalizationLayer();
        FeatureMap[,] gradient = new FeatureMap[,] { { new FeatureMap(3, 3, new Color(1)) } };
        testOutput = layer.Forward(testInput);

        FeatureMap output = new FeatureMap(3, 3);
        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                output[i, j] = testOutput[0, 0][i, j];
            }
        }
        
        FeatureMap[,] outGradient = layer.Backwards(testInput, gradient, 0);

        float h = 0.0001f;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Color hColor = k switch
                    {
                        0 => new Color(h, 0, 0),
                        1 => new Color(0, h, 0),
                        2 => new Color(0, 0, h)
                    };
                    testInput[0, 0][i, j] += hColor;
                    testOutput = layer.Forward(testInput);
                    float testGradient = 0;
                    for (int i2 = 0; i2 < 3; i2++)
                    {
                        for (int j2 = 0; j2 < 3; j2++)
                        {
                            for (int k2 = 0; k2 < 3; k2++)
                            {
                                testGradient += (testOutput[0, 0][i2, j2][k2] - output[i2, j2][k2]) / h;
                            }
                        }
                    }
                    Console.WriteLine($"Expected Gradient: {outGradient[0, 0][i, j][k]:f4} \t Test Gradient: {testGradient:f4}");
                    testInput[0, 0][i, j] -= hColor;

                }
            }
        }
    }

    public static void TestFullyConnectedLayer()
    {
        FeatureMap[,] testInput = new FeatureMap[,] { { new FeatureMap(3, 3) } };
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                testInput[0, 0][i, j] = new Color(i, j, i - j);
            }
        }
        FeatureMap[,] testOutput = testInput;
        FullyConnectedLayer layer = new FullyConnectedLayer(1);
        FeatureMap[,] gradient = new FeatureMap[,] { { new FeatureMap(3, 3, new Color(1)) } };
        testOutput = layer.Forward(testInput);

        FeatureMap output = new FeatureMap(3, 3);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                output[i, j] = testOutput[0, 0][i, j];
            }
        }

        FeatureMap[,] outGradient = layer.Backwards(testInput, gradient, 0);

        float h = 0.0001f;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Color hColor = k switch
                    {
                        0 => new Color(h, 0, 0),
                        1 => new Color(0, h, 0),
                        2 => new Color(0, 0, h)
                    };
                    testInput[0, 0][i, j] += hColor;
                    testOutput = layer.Forward(testInput);
                    float testGradient = (testOutput[0, 0][i, j][0] - output[i, j][0]) / h;
                    testGradient += (testOutput[0, 0][i, j][1] - output[i, j][1]) / h;
                    testGradient += (testOutput[0, 0][i, j][2] - output[i, j][2]) / h;
                    Console.WriteLine($"Expected Gradient: {outGradient[0, 0][i, j][k]:f4} \t Test Gradient: {testGradient:f4}");
                    testInput[0, 0][i, j] -= hColor;

                }
            }
        }
    }

    public static void TestFullyConnectedMultipliers()
    {
        FeatureMap[,] testInput = new FeatureMap[,] { { new FeatureMap(3, 3) } };
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                testInput[0, 0][i, j] = new Color(i, j, i - j);
            }
        }
        FeatureMap[,] testOutput = testInput;
        FullyConnectedLayer layer = new FullyConnectedLayer(1);
        FeatureMap[,] gradient = new FeatureMap[,] { { new FeatureMap(3, 3, new Color(1)) } };
        testOutput = layer.Forward(testInput);

        FeatureMap output = new FeatureMap(3, 3);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                output[i, j] = testOutput[0, 0][i, j];
            }
        }
        layer.Backwards(testInput, gradient, 0);

        float h = 0.0001f;


/*        for (int k = 0; k < 9; k++)
        {
            Color hColor = (k % 3) switch
            {
                0 => new Color(h, 0, 0),
                1 => new Color(0, h, 0),
                2 => new Color(0, 0, h)
            };
            switch (k / 3)
            {
                case 0:
                    layer._redMatrix[0, 0] += hColor;
                    break;
                case 1:
                    layer._greenMatrix[0, 0] += hColor;
                    break;
                case 2:
                    layer._blueMatrix[0, 0] += hColor;
                    break;
            }
            testOutput = layer.Forward(testInput);
            float testGradient = 0;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    testGradient += (testOutput[0, 0][i, j][0] - output[i, j][0]) / h;
                    testGradient += (testOutput[0, 0][i, j][1] - output[i, j][1]) / h;
                    testGradient += (testOutput[0, 0][i, j][2] - output[i, j][2]) / h;
                }
            }
            Console.WriteLine($"Expected Gradient: {layer.Gradients[0, 0][k]:f4} \t Test Gradient: {testGradient:f4}");
            switch (k / 3)
            {
                case 0:
                    layer._redMatrix[0, 0] -= hColor;
                    break;
                case 1:
                    layer._greenMatrix[0, 0] -= hColor;
                    break;
                case 2:
                    layer._blueMatrix[0, 0] -= hColor;
                    break;
            }
        }*/
    }
}

