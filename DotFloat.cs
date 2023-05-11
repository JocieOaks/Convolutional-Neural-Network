// See https://aka.ms/new-console-template for more information

using Newtonsoft.Json;

[Serializable]
public readonly struct DotFloat : IDot<DotFloat>
{
    public float Value { get; }

    public DotFloat Multiply(DotFloat other)
    {
        return Multiply(other.Value);
    }

    public DotFloat Random()
    {
        return new DotFloat((float)CLIP.Random.NextDouble());
    }

    public DotFloat Add(DotFloat other)
    {
        return new DotFloat(Value + other.Value);
    }

    public DotFloat Multiply(float multiple)
    {
        return new DotFloat(Value * multiple);
    }

    public DotFloat Divide(DotFloat other)
    {
        return new DotFloat(Value / other.Value);
    }

    public DotFloat ReLU()
    {
        if(Value < 0)
            return new DotFloat(0);
        return this;
    }

    public DotFloat Subtract(DotFloat other)
    {
        return new DotFloat(Value - other.Value);
    }

    public DotFloat Pow(float power)
    {
        return new DotFloat(MathF.Pow(Value, power));
    }

    public DotFloat Add(float value)
    {
        return new DotFloat(Value + value);
    }

    [JsonConstructor]
    public DotFloat(float value)
    {
        Value = value;
    }

    public DotFloat()
    {
        Value = 0;
    }

    public static implicit operator DotFloat(float value)
    {
        return new DotFloat(value);
    }
}
