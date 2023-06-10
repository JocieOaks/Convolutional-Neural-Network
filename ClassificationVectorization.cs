public static class ClassificationVectorization
{
    public static Vector Vectorize(bool[] bools, float[] floats)
    {
        Vector vector = new Vector(bools.Length + floats.Length);
        for(int i = 0; i < bools.Length; i++)
        {
            vector[i] = bools[i] ? 1 : -1;
        }
        for(int i =0; i < floats.Length; i++)
        {
            vector[bools.Length + i] = floats[i] * 2 - 1;
        }

        return vector.Normalized();
    }

}
