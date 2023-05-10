// See https://aka.ms/new-console-template for more information
public struct DotFloat : IDot<DotFloat>
{
    public float Value { get; }

    public DotFloat Multiply(DotFloat other)
    {
        return Multiply(other.Value);
    }

    public DotFloat Random()
    {
        return new DotFloat((float)new Random().NextDouble());
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

    public DotFloat(float value)
    {
        Value = value;
    }

    public DotFloat()
    {
        Value = 0;
    }
}
