// See https://aka.ms/new-console-template for more information

public enum NormalizationLayers
{
    Activation,
    BatchNormalization,
    Dropout
}
public struct ActivationPattern
{
    NormalizationLayers[] _pattern;
    float _dropoutRate;

    public ActivationPattern(NormalizationLayers[] pattern, float dropoutRate)
    {
        _pattern = pattern;
        _dropoutRate = dropoutRate;
    }

    public IEnumerable<ISecondaryLayer> GetLayers()
    {
        foreach(var layer in  _pattern)
        {
            yield return layer switch
            {
                NormalizationLayers.Activation => new ReLULayer(),
                NormalizationLayers.BatchNormalization => new BatchNormalizationLayer(),
                NormalizationLayers.Dropout => new DropoutLayer(_dropoutRate)
            };
        }
    }
}