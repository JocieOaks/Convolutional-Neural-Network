﻿using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.GPU;
using ILGPU;
using ILGPU.Runtime;
using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace ConvolutionalNeuralNetwork.Layers.Skip
{
    /// <summary>
    /// The <see cref="SkipConcatenate"/> class is a <see cref="Layer"/> for combining a set of <see cref="FeatureMap"/>s from the previous
    /// <see cref="Layer"/> with the <see cref="FeatureMap"/>s from its corresponding <see cref="SkipSplit"/>.
    /// </summary>
    public class SkipOut : Layer, IStructuralLayer, ISkipEndpoint
    {
        private Vector _skipConnection;
        private Shape _skipShape;

        [JsonProperty] public int ID { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="SkipConcatenate"/> class.
        /// </summary>
        public SkipOut() : base(1, 1)
        {
        }

        [OnDeserialized]
        private void OnDeserialized(StreamingContext context)
        {
            if(SkipSplit.Splits.TryGetValue(ID, out var split))
            {
                split.Connect(this);
            }
            else
            {
                throw new Exception("Split source cannot be found.");
            }
        }

        /// <inheritdoc/>
        public override string Name => "Skip Out Layer";

        /// <inheritdoc/>
        public override void Backwards(int batchSize)
        {
            Index1D index = new(batchSize * _skipShape.Volume);
            GPUManager.CopyAction(index, _buffers.InGradient, _skipConnection.GetArrayViewEmpty<float>());

            Synchronize();
            _skipConnection.DecrementLiveCount();
        }

        /// <summary>
        /// Connects the <see cref="SkipConcatenate"/> with its <see cref="SkipSplit"/> sharing the <see cref="FeatureMap"/>s
        /// between them.
        /// </summary>
        /// <param name="inputs">The split outputs of the <see cref="SkipSplit"/>.</param>
        /// <param name="outGradients">The split inGradients of the <see cref="SkipSplit"/>.</param>
        public void Connect(Vector skipConnection, Shape skipInputShape, int id)
        {
            _skipConnection = skipConnection;
            _skipShape = skipInputShape;
            ID = id;
        }

        /// <inheritdoc/>
        public override void Forward(int batchSize)
        {
            Index1D index = new(batchSize * _skipShape.Volume);
            GPUManager.CopyAction(index, _skipConnection.GetArrayViewEmpty<float>(), _buffers.Output);

            Synchronize();
            _skipConnection.DecrementLiveCount();
        }

        /// <inheritdoc/>
        public override void Reset()
        {
        }

        /// <inheritdoc/>
        public override Shape Startup(Shape inputShape, IOBuffers buffers, int batchSize)
        {
            if (_ready)
                return _outputShape;
            _ready = true;

            _outputShape = _skipShape;

            _buffers = buffers;

            buffers.OutputDimensionArea(_outputShape.Volume);
            return _outputShape;
        }
    }
}