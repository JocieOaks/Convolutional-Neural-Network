namespace LayerShape
{
    public interface ILayerShape
    {
        (int, int) OutputResolution((int width, int length) input);
        (int, int) InputResolution((int width, int length) output);
        int? OutputDimensions(int inputDimensions);
        int? InputDimensions(int outputDimensions);
        IPrimaryLayer Create();
    }

    public readonly struct ConvolutionalShape : ILayerShape
    {
        public IPrimaryLayer Create()
        {
            if (Key)
                return new ConvolutionalKeyLayer(FilterSize, Stride, DimensionMultiplier.HasValue ? DimensionMultiplier.Value : 1);
            return new ConvolutionalLayer(FilterSize, Stride, DimensionMultiplier.HasValue ? DimensionMultiplier.Value : 1);
        }

        public int? InputDimensions(int outputDimensions)
        {
            if (!DimensionMultiplier.HasValue)
                return null;
            if (DimensionMultiplier > 0)
            {
                return outputDimensions % DimensionMultiplier != 0 ? null : outputDimensions / DimensionMultiplier;
            }
            else
            {
                return outputDimensions * -DimensionMultiplier;
            }
        }

        public (int, int) InputResolution((int width, int length) output)
        {
            int inputWidth = Stride * (output.width - 2) + 1 + FilterSize;
            int inputLength = Stride * (output.length - 2) + 1 + FilterSize;

            return (inputWidth, inputLength);
        }

        public int? OutputDimensions(int inputDimensions)
        {
            if (!DimensionMultiplier.HasValue)
                return null;
            if (DimensionMultiplier > 0)
            {
                return inputDimensions * DimensionMultiplier;
            }
            else
            {
                return inputDimensions % -DimensionMultiplier != 0 ? null : inputDimensions / -DimensionMultiplier;
            }
        }

        public (int, int) OutputResolution((int width, int length) input)
        {
            int outputWidth = 2 + (input.width - FilterSize - 1) / Stride;
            int outputLength = 2 + (input.length - FilterSize - 1) / Stride;
            return (outputWidth, outputLength);
        }

        public int FilterSize { get; init; }
        public int Stride { get; init; }
        public int? DimensionMultiplier { get; init; }
        public bool Key { get; init; }
    }

    public readonly struct PoolShape : ILayerShape
    {
        public IPrimaryLayer Create()
        {
            return new AveragePoolLayer(FilterSize);
        }

        public int? InputDimensions(int outputDimensions)
        {
            return outputDimensions;
        }

        public (int, int) InputResolution((int width, int length) output)
        {
            return (output.width * FilterSize, output.length * FilterSize);
        }

        public int? OutputDimensions(int inputDimensions)
        {
            return inputDimensions;
        }

        public (int, int) OutputResolution((int width, int length) input)
        {
            return (input.width / FilterSize, input.length / FilterSize);
        }

        public int FilterSize { get; init; }
    }

    public readonly struct FullyConnectedShape : ILayerShape
    {
        public IPrimaryLayer Create()
        {
            FullyConnectedLayer layer = new FullyConnectedLayer();
            if(DimensionMultiplier.HasValue)
                layer.SetOutputMultiplier(DimensionMultiplier.Value);
            return layer;
        }

        public int? InputDimensions(int outputDimensions)
        {
            if (!DimensionMultiplier.HasValue)
                return null;
            if (DimensionMultiplier > 0)
            {
                return outputDimensions % DimensionMultiplier != 0 ? null : outputDimensions / DimensionMultiplier;
            }
            else
            {
                return outputDimensions * -DimensionMultiplier;
            }
        }

        public (int, int) InputResolution((int width, int length) output)
        {
            return (output.width,  output.length);
        }

        public int? OutputDimensions(int inputDimensions)
        {
            if (!DimensionMultiplier.HasValue)
                return null;
            if (DimensionMultiplier > 0)
            {
                return inputDimensions * DimensionMultiplier;
            }
            else
            {
                return inputDimensions % -DimensionMultiplier != 0 ? null : inputDimensions / -DimensionMultiplier;
            }
        }

        public (int, int) OutputResolution((int width, int length) input)
        {
            return (input.width, input.length);
        }

        public int? DimensionMultiplier { get; init; }
    }

    public readonly struct ScalingShape : ILayerShape
    {
        public IPrimaryLayer Create()
        {
            ScalingLayer scalingLayer = new ScalingLayer();
            scalingLayer.SetScale(ScalingMultiplier, ScalingMultiplier);
            return scalingLayer;
        }

        public int? InputDimensions(int outputDimensions)
        {
            throw new NotImplementedException();
        }

        public (int, int) InputResolution((int width, int length) output)
        {
            throw new NotImplementedException();
        }

        public int? OutputDimensions(int inputDimensions)
        {
            throw new NotImplementedException();
        }

        public (int, int) OutputResolution((int width, int length) input)
        {
            throw new NotImplementedException();
        }

        public int ScalingMultiplier { get; init; }

    }
}