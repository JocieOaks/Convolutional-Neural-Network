using ConvolutionalNeuralNetwork.Design.LayerBlueprints;
using ConvolutionalNeuralNetwork.Layers;
using ConvolutionalNeuralNetwork.Networks;
using System.Collections.ObjectModel;

namespace ConvolutionalNeuralNetwork.Design
{
    public readonly struct UNetConstructor
    {
        public ReadOnlyCollection<ILayerBlueprint> DownPattern { get; init; }

        public ReadOnlyCollection<ILayerBlueprint> UpPattern { get; init; }

        public ReadOnlyCollection<ILayerBlueprint> ValleyPattern { get; init; }

        public ReadOnlyCollection<ILayerBlueprint> EntrancePattern { get; init; }

        public ReadOnlyCollection<ILayerBlueprint> ExitPattern { get; init; }

        public int? DownDimensionMultiplier { get; init; }

        public int? UpDimensionMultiplier { get; init; }

        public readonly ILayerBlueprint DownScaler { get; init; } = new PoolBlueprint() { FilterSize = 2 };

        public UNetConstructor()
        { }

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

            IEnumerable<ILayerBlueprint> downPattern = GetDownPattern();

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
                Scaling scalingLayer = new();
                scalingLayer.SetDimensions(outputResolution.width, outputResolution.length);
                generator.AddLayer(scalingLayer);
            }

            return generator;
        }

        private Generator GetLayers(IEnumerable<ILayerBlueprint> downPattern, int levels, ref (int width, int length) resolution)
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
                Scaling scalingLayer = new();
                scalingLayer.SetDimensions(resolution.width, resolution.length);
                generator.AddLayer(scalingLayer);
                index++;
                generator.AddSkipConnection(skipConnections.Pop(), index);
                index += 2;
                foreach (var layer in GetUpLayers(dimensions, dimensions += levelDimensions[i]))
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
            if (dimensions != 1)
            {
                FullyConnected fullyConnectedLayer = new();
                fullyConnectedLayer.SetOutputDimensions(1);
                generator.AddLayer(fullyConnectedLayer);
            }
            return generator;
        }

        private IEnumerable<ILayerBlueprint> GetUpLayers(int inputDimensions, int concatDimensions)
        {
            List<ILayerBlueprint> upPattern = new();

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

                    if (dimensions != targetDimensions && unsetLayers == 0 || dimensions % targetDimensions != 0)
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
                            if (layer is ConvolutionBlueprint conShape && conShape.DimensionMultiplier == null)
                            {
                                upPattern.Add(new ConvolutionBlueprint()
                                {
                                    FilterSize = conShape.FilterSize,
                                    Stride = conShape.Stride,
                                    DimensionMultiplier = -primeScalings[index++]
                                });
                            }
                            else if (layer is FullyConnectedBlueprint fulShape && fulShape.DimensionMultiplier == null)
                            {
                                upPattern.Add(new FullyConnectedBlueprint()
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
                    return new List<ILayerBlueprint>() { new FullyConnectedBlueprint()
                {
                    DimensionMultiplier = UpDimensionMultiplier
                }
                };
                }
            }
            else if (UpPattern != null)
            {
                if (UpPattern.Any(x => x is ConvolutionBlueprint conShape && conShape.DimensionMultiplier == null || x is FullyConnectedBlueprint fulShape && fulShape.DimensionMultiplier == null))
                {
                    foreach (var layer in UpPattern)
                    {
                        if (layer is ConvolutionBlueprint conShape && conShape.DimensionMultiplier == null)
                        {
                            upPattern.Add(new ConvolutionBlueprint()
                            {
                                FilterSize = conShape.FilterSize,
                                Stride = conShape.Stride,
                                DimensionMultiplier = 1
                            });
                        }
                        else if (layer is FullyConnectedBlueprint fulShape && fulShape.DimensionMultiplier == null)
                        {
                            upPattern.Add(new FullyConnectedBlueprint()
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

        private IEnumerable<ILayerBlueprint> GetDownPattern()
        {
            List<ILayerBlueprint> downPattern = new();

            if (DownDimensionMultiplier != null)
            {
                if (DownPattern != null)
                {
                    int dimensions = 1;
                    int unsetLayers = 0;
                    foreach (var layer in DownPattern)
                    {
                        int? nextDimension = layer.OutputDimensions(dimensions);
                        if (nextDimension == null)
                            unsetLayers++;
                        else
                            dimensions = nextDimension.Value;
                    }

                    if (dimensions != DownDimensionMultiplier && unsetLayers == 0 || DownDimensionMultiplier % dimensions != 0)
                    {
                        throw new ArgumentException("Constraints can not be fulfilled");
                    }

                    if (unsetLayers == 0)
                    {
                        return DownPattern;
                    }
                    else
                    {
                        int[] primeScalings = PrimeFactorization(DownDimensionMultiplier.Value / dimensions, unsetLayers);
                        int index = 0;

                        foreach (var layer in DownPattern)
                        {
                            if (layer is ConvolutionBlueprint conShape && conShape.DimensionMultiplier == null)
                            {
                                downPattern.Add(new ConvolutionBlueprint()
                                {
                                    FilterSize = conShape.FilterSize,
                                    Stride = conShape.Stride,
                                    DimensionMultiplier = primeScalings[index++]
                                });
                            }
                            else if (layer is FullyConnectedBlueprint fulShape && fulShape.DimensionMultiplier == null)
                            {
                                downPattern.Add(new FullyConnectedBlueprint()
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
                    return new List<ILayerBlueprint>() { new ConvolutionBlueprint()
                {
                    FilterSize = 3,
                    Stride = 1,
                    DimensionMultiplier = DownDimensionMultiplier
                }
                };
                }
            }
            else if (DownPattern != null)
            {
                if (DownPattern.Any(x => x is ConvolutionBlueprint conShape && conShape.DimensionMultiplier == null || x is FullyConnectedBlueprint fulShape && fulShape.DimensionMultiplier == null))
                {
                    foreach (var layer in DownPattern)
                    {
                        if (layer is ConvolutionBlueprint conShape && conShape.DimensionMultiplier == null)
                        {
                            downPattern.Add(new ConvolutionBlueprint()
                            {
                                FilterSize = conShape.FilterSize,
                                Stride = conShape.Stride,
                                DimensionMultiplier = 1
                            });
                        }
                        else if (layer is FullyConnectedBlueprint fulShape && fulShape.DimensionMultiplier == null)
                        {
                            downPattern.Add(new FullyConnectedBlueprint()
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
            for (int i = 0; i < factorCount; i++)
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
}