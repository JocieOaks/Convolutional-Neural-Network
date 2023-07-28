using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.DataTypes
{
    public readonly struct Shape
    {
        public int Dimensions { get; }
        public int Width { get; }

        public int Length { get; }

        public int Area => Width * Length;

        public int Volume => Area * Dimensions;

        public Shape(int width, int length, int dimensions)
        {
            Width = width;
            Length = length;
            Dimensions = dimensions;
        }

        public int FromCoordinates(int x, int y)
        {
            return Width * y + x;
        }

        public (int, int) FromIndex(int index)
        {
            int x = index % Width;
            int y = index / Width;
            return (x, y);
        }

        public int GetOffset(int batchIndex, int dimension)
        {
            return (batchIndex * Dimensions + dimension) * Area;
        }

        public bool TryGetIndex(int indexIndex, int shiftX, int shiftY, out int outIndex)
        {

            int y = indexIndex / Width;
            int x = indexIndex - (y * Width);

            shiftX += x;
            shiftY += y;
            outIndex = shiftY * Width + shiftX;
            return shiftX >= 0 && shiftY >= 0 && shiftX < Width && shiftY < Length;
        }
    }
}
