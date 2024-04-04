# Convolutional Neural Network
A simple library for creating neural networks in C#.
## Description
This is a usable library for creating convolutional neural networks. However, this project began simply as a learning exercise to study how neural networks are implemented. Therefore although an effort has been made to make the library as efficient as possible, it is not necessarily the most optomized implementation. The primary goal was simply some hands on experimentation.

The library runs GPU kernals using ILGPU to perform layer calculations quickly.
## Requirements
* [ILGPU](https://ilgpu.net/)
* ILGPU.Algorithms
* [Newtonsoft.JSON](https://www.newtonsoft.com/json)
* System.Drawing.Common (only needed for the example)
## Usage
The Network class is used to create and train new neural networks, and trained on images converted into Tensor form. Layers are added to the Network by using the "AddLayer" methods. The Example folder provides utilities for converting between Windows Bitmap and Tensors.

Custom Layers can be created by inheriting from the Layer class and then creating the corresponding SerialLayer (for layers with weights, inherit from WeightedLayer and SerialWeighted).
### Example
The example gives a basic implementation for a GAN that trains on a single handwritten symbol. It was built for the MNIST data set, but other data sets should also be usable. The SymbolGAN class works on any platform, but the TrainSymbol class gives out-of-the-box implementation for Windows. For other operating systems, conversion from images to Tensors will have to be custom made.
## License
This project is under an MIT license.
