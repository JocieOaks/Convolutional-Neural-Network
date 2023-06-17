using LayerShape;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Configuration;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public readonly struct UNetConstructor
{
    public ReadOnlyCollection<ILayerShape> DownPattern { get; init; }

    public ReadOnlyCollection<ILayerShape> UpPattern { get; init; }

    public ReadOnlyCollection<ILayerShape> ValleyPattern { get; init; }

    public ReadOnlyCollection<ILayerShape> EntrancePattern { get; init; }

    public ReadOnlyCollection<ILayerShape> ExitPattern { get; init; }

    public int? DownDimensionMultiplier { get; init; }

    public int? UpDimensionMultiplier { get; init; }

    private readonly ILayerShape DownScaler { get; } = new PoolShape() { FilterSize = 2 };

    public UNetConstructor() { }

    public Generator BuildUNet((int width, int length) inputResolution, (int width, int length) outputResolution = default, int maxLevels = int.MaxValue)
    {

        if (EntrancePattern != null)
        {
            foreach (var layer in EntrancePattern)
            {
                inputResolution = layer.OutputResolution(inputResolution);
            }
        }

        (int width, int length) minSize = (3, 3);

        if (ValleyPattern != null)
        {
            foreach (var layer in ValleyPattern.Reverse())
            {
                minSize = layer.InputResolution(minSize);
            }
        }

        IEnumerable<ILayerShape> downPattern = GetDownPattern();

        int levels = 0;
        while (levels < maxLevels && minSize.width < inputResolution.width && minSize.length < inputResolution.length)
        {
            levels++;
            foreach (var layer in downPattern.Reverse())
            {
                minSize = layer.InputResolution(minSize);
            }
            minSize = DownScaler.InputResolution(minSize);
        }


        Generator generator = GetLayers(downPattern, levels, ref inputResolution);

        if (outputResolution != default && inputResolution != outputResolution)
        {
            ScalingLayer scalingLayer = new();
            scalingLayer.SetDimensions(outputResolution.width, outputResolution.length);
            generator.AddLayer(scalingLayer);
        }

        return generator;
    }

    Generator GetLayers(IEnumerable<ILayerShape> downPattern, int levels, ref (int width, int length) resolution)
    {
        Generator generator = new();
        Stack<int> skipConnections = new();
        Stack<(int, int)> skipResolutions = new();
        int index = 0;
        int[] levelDimensions = new int[levels];
        int dimensions = 1;
        if (EntrancePattern != null)
        {
            foreach (var layer in EntrancePattern)
            {
                dimensions = layer.OutputDimensions(dimensions).Value;
                generator.AddLayer(layer.Create());
                index++;
            }
        }

        for (int i = 0; i < levels; i++)
        {
            foreach (var layer in downPattern)
            {
                dimensions = layer.OutputDimensions(dimensions).Value;
                resolution = layer.OutputResolution(resolution);
                generator.AddLayer(layer.Create());
                index++;
            }
            skipConnections.Push(index);
            skipResolutions.Push(resolution);

            resolution = DownScaler.OutputResolution(resolution);
            generator.AddLayer(DownScaler.Create());
            index++;

            levelDimensions[i] = dimensions;
        }

        if (ValleyPattern != null)
        {
            foreach (var layer in ValleyPattern)
            {
                dimensions = layer.OutputDimensions(dimensions).Value;
                resolution = layer.OutputResolution(resolution);
                generator.AddLayer(layer.Create());
                index++;
            }
        }

        for (int i = levels - 1; i >= 0; i--)
        { 
            resolution = skipResolutions.Pop();
            ScalingLayer scalingLayer = new ScalingLayer();
            scalingLayer.SetDimensions(resolution.width, resolution.length);
            generator.AddLayer(scalingLayer);
            index++;
            generator.AddSkipConnection(skipConnections.Pop(), index);
            index += 2;
            foreach(var layer in GetUpLayers(dimensions, dimensions += levelDimensions[i]))
            {
                dimensions = layer.OutputDimensions(dimensions).Value;
                resolution = layer.OutputResolution(resolution);
                generator.AddLayer(layer.Create());
                index++;
            }
        }

        if (ExitPattern != null)
        {
            foreach (var layer in ExitPattern)
            {
                dimensions = layer.OutputDimensions(dimensions).Value;
                resolution = layer.OutputResolution(resolution);
                generator.AddLayer(layer.Create());
            }
        }
        if(dimensions != 1)
        {
            FullyConnectedLayer fullyConnectedLayer = new();
            fullyConnectedLayer.SetOutputDimensions(1);
            generator.AddLayer(fullyConnectedLayer);
        }
        return generator;
    }

    IEnumerable<ILayerShape> GetUpLayers(int inputDimensions, int concatDimensions)
    {
        List<ILayerShape> upPattern = new List<ILayerShape>();

        if (UpDimensionMultiplier != null)
        {
            if (inputDimensions % -UpDimensionMultiplier != 0)
                throw new ArgumentException("Constraints cannot be fulfilled.");

            int targetDimensions = inputDimensions / -UpDimensionMultiplier.Value;

            if (UpPattern != null)
            {
                int dimensions = concatDimensions;
                int unsetLayers = 0;
                foreach (var layer in UpPattern)
                {
                    int? nextDimension = layer.OutputDimensions(dimensions);
                    if (nextDimension == null)
                        unsetLayers++;
                    else
                        dimensions = nextDimension.Value;
                }

                if ((dimensions != targetDimensions && unsetLayers == 0) || dimensions % targetDimensions != 0)
                {
                    throw new ArgumentException("Constraints cannot be fulfilled.");
                }

                if (unsetLayers == 0)
                {
                    return UpPattern;
                }
                else
                {
                    int[] primeScalings = PrimeFactorization(dimensions / targetDimensions, unsetLayers);
                    int index = 0;

                    foreach (var layer in UpPattern)
                    {
                        if (layer is ConvolutionalShape conShape && conShape.DimensionMultiplier == null)
                        {
                            upPattern.Add(new ConvolutionalShape()
                            {
                                FilterSize = conShape.FilterSize,
                                Stride = conShape.Stride,
                                DimensionMultiplier = -primeScalings[index++]
                            });
                        }
                        else if (layer is FullyConnectedShape fulShape && fulShape.DimensionMultiplier == null)
                        {
                            upPattern.Add(new FullyConnectedShape()
                            {
                                DimensionMultiplier = -primeScalings[index++]
                            });
                        }
                        else
                            upPattern.Add(layer);
                    }
                    return upPattern;
                }
            }
            else
            {
                return new List<ILayerShape>() { new FullyConnectedShape()
                {
                    DimensionMultiplier = UpDimensionMultiplier
                }
                };
            }
        }
        else if (UpPattern != null)
        {
            if (UpPattern.Any(x => (x is ConvolutionalShape conShape && conShape.DimensionMultiplier == null) || (x is FullyConnectedShape fulShape && fulShape.DimensionMultiplier == null)))
            {
                foreach (var layer in UpPattern)
                {
                    if (layer is ConvolutionalShape conShape && conShape.DimensionMultiplier == null)
                    {
                        upPattern.Add(new ConvolutionalShape()
                        {
                            FilterSize = conShape.FilterSize,
                            Stride = conShape.Stride,
                            DimensionMultiplier = 1
                        });
                    }
                    else if (layer is FullyConnectedShape fulShape && fulShape.DimensionMultiplier == null)
                    {
                        upPattern.Add(new FullyConnectedShape()
                        {
                            DimensionMultiplier = 1
                        });
                    }
                    else
                        upPattern.Add(layer);
                }
                return upPattern;
            }
            else
            {
                return UpPattern;
            }
        }
        else
        {
            return upPattern;
        }
    }

    IEnumerable<ILayerShape> GetDownPattern()
    {
        List<ILayerShape> downPattern = new List<ILayerShape>();

        if(DownDimensionMultiplier != null)
        {
            if(DownPattern != null)
            {
                int dimensions = 1;
                int unsetLayers = 0;
                foreach(var layer in DownPattern)
                {
                    int? nextDimension = layer.OutputDimensions(dimensions);
                    if (nextDimension == null)
                        unsetLayers++;
                    else
                        dimensions = nextDimension.Value;
                }

                if((dimensions != DownDimensionMultiplier && unsetLayers == 0) || DownDimensionMultiplier % dimensions != 0)
                {
                    throw new ArgumentException("Constraints can not be fulfilled");
                }
                
                if(unsetLayers == 0)
                {
                    return DownPattern;
                }
                else
                {
                    int[] primeScalings = PrimeFactorization(DownDimensionMultiplier.Value / dimensions, unsetLayers);
                    int index = 0;

                    foreach (var layer in DownPattern)
                    {
                        if (layer is ConvolutionalShape conShape && conShape.DimensionMultiplier == null)
                        {
                            downPattern.Add(new ConvolutionalShape()
                            {
                                FilterSize = conShape.FilterSize,
                                Stride = conShape.Stride,
                                DimensionMultiplier = primeScalings[index++]
                            });
                        }
                        else if (layer is FullyConnectedShape fulShape && fulShape.DimensionMultiplier == null)
                        {
                            downPattern.Add(new FullyConnectedShape()
                            {
                                DimensionMultiplier = primeScalings[index++]
                            });
                        }
                        else
                            downPattern.Add(layer);
                    }
                    return downPattern;
                }
            }
            else
            {
                return new List<ILayerShape>() { new ConvolutionalShape()
                {
                    FilterSize = 3,
                    Stride = 1,
                    DimensionMultiplier = DownDimensionMultiplier
                }
                };
            }
        }
        else if(DownPattern != null)
        {
            if(DownPattern.Any(x => (x is ConvolutionalShape conShape && conShape.DimensionMultiplier == null)|| (x is FullyConnectedShape fulShape && fulShape.DimensionMultiplier == null)))
            {
                foreach (var layer in DownPattern)
                {
                    if (layer is ConvolutionalShape conShape && conShape.DimensionMultiplier == null)
                    {
                        downPattern.Add(new ConvolutionalShape()
                        {
                            FilterSize = conShape.FilterSize,
                            Stride = conShape.Stride,
                            DimensionMultiplier = 1
                        });
                    }
                    else if (layer is FullyConnectedShape fulShape && fulShape.DimensionMultiplier == null)
                    {
                        downPattern.Add(new FullyConnectedShape()
                        {
                            DimensionMultiplier = 1
                        });
                    }
                    else
                        downPattern.Add(layer);
                }
                return downPattern;
            }
            else
            {
                return DownPattern;
            }
        }
        else
        {
            return downPattern;
        }
    }

    public static int[] PrimeFactorization(int integer, int factorCount)
    {
        if (factorCount == 1)
            return new int[] { integer };

        int[] factors = new int[factorCount];
        for(int i = 0; i < factorCount; i++)
        {
            factors[i] = 1;
        }
        while (GetNextFactor()) { }
        AddFactor(integer);
        return factors;

        bool GetNextFactor()
        {
            for (int i = 2; i < integer; i++)
            {
                if (integer % 2 == 0)
                {
                    AddFactor(i);
                    integer /= i;
                    return true;
                }
            }
            return false;
        }

        void AddFactor(int factor)
        {
            factors[1] = factors[1] * factors[0];
            factors[0] = factor;
            Quicksort(0, factorCount - 1);
        }

        void Quicksort(int low, int high)
        {
            if (high < low)
                return;
            int pivot = factors[high];
            int spot = low;
            for (int i = low; i < high; i++)
            {
                if (factors[i] > pivot)
                {
                    (factors[spot], factors[i]) = (factors[i], factors[spot]);
                    spot++;
                }
            }
            (factors[spot], factors[high]) = (factors[high], factors[spot]);

            Quicksort(low, spot - 1);
            Quicksort(spot + 1, high);
        }
    }
}