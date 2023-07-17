using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    public readonly struct Shape
    {
        public int Width { get; }

        public int Length { get; }

        public int Area => Width * Length;

        public Shape(int width, int length)
        {
            Width = width;
            Length = length;
        }
    }
}
