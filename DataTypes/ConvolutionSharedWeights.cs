using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    public class ConvolutionSharedWeights : SharedWeights
    {
        public int FilterSize { get; }
        public int Stride { get; }
        public int Dimensions { get; }

        public ConvolutionSharedWeights(int filterSize, int stride, int dimensions)
        {
            FilterSize = filterSize;
            Stride = stride;
            Dimensions = dimensions;
        }
    }
}
