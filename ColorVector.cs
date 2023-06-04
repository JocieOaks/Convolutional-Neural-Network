using Newtonsoft.Json;

[Serializable]
public class ColorVector
{
    [JsonProperty] private readonly Color[] _values;

    public ColorVector(Color[] values)
    {
        _values = values;
    }

    public ColorVector(int length)
    {
        _values = new Color[length];
    }

    [JsonConstructor] private ColorVector() { }

    [JsonIgnore] public int Length => _values.Length;

    public Color this[int index]
    {
        get => _values[index];
        set => _values[index] = value;
    }

    public static Vector operator *(FeatureMap matrix, ColorVector vector)
    {
        if (matrix.Length != vector.Length)
            throw new ArgumentException("Matrix and vector are not compatible.");

        Vector output = new(matrix.Width);
        for (int i = 0; i < matrix.Width; i++)
        {
            for (int j = 0; j < matrix.Length; j++)
            {
                output[i] += Color.Dot(matrix[i, j], vector[j]);
            }
        }

        return output;
    }

    public static ColorVector operator *(ColorVector vector, FeatureMap matrix)
    {
        if (matrix.Width != vector.Length)
            throw new ArgumentException("Matrix and vector are not compatible.");
        ColorVector output = new(matrix.Length);
        for (int i = 0; i < matrix.Width; i++)
        {
            for (int j = 0; j < matrix.Length; j++)
            {
                output[j] += matrix[i, j] * vector[i];
            }
        }

        return output;
    }

    public Vector Magnitude()
    {
        Vector vector = new Vector(Length);
        for (int i = 0; i < Length; i++)
        {
            vector[i] = _values[i].Magnitude;
        }
        return vector;
    }
}