// See https://aka.ms/new-console-template for more information
public class VectorizationLayer
{
    int _vectorDimensions;
    float[] _weight;
    float[] _bias;

    public VectorizationLayer(int vectorDimensions)
    {
        _vectorDimensions = vectorDimensions;
        _weight = new float[vectorDimensions];
        for(int i = 0; i < vectorDimensions; i++)
        {
            _weight[i] = 1;
        }
        _bias = new float[vectorDimensions];
    }

    public Vector Forward(DotFloat[][,] input) 
    { 
        float[] vector = new float[_vectorDimensions];
        float[] weightedVector = new float[_vectorDimensions];

        int position = 0;

        foreach(var featuremap in input)
        {
            for(int j = 0; j < featuremap.GetLength(0); j++)
            {
                for(int k = 0; k < featuremap.GetLength(1); k++)
                {
                    vector[position] += featuremap[j, k].Value;
                    position = (position + 1) % _vectorDimensions;
                }
            }
        }

        for(int i = 0; i < _vectorDimensions; i++)
        {
            weightedVector[i] = vector[i] * _weight[i] + _bias[i];
        }
        return new Vector(weightedVector);
    }

    public DotFloat[][,] Backwards(float[] error)
    {
        throw new NotImplementedException();
    }
}
