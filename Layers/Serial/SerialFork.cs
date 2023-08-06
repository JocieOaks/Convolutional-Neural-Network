﻿using ConvolutionalNeuralNetwork.DataTypes;
using ConvolutionalNeuralNetwork.Layers.SkipConnection;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork.Layers.Serial
{
    public class SerialFork : ISerial
    {
        private static int s_nextID = 0;

        [JsonIgnore] public static readonly Dictionary<int, Fork> Forks = new();

        public SerialFork()
        {
            ID = s_nextID++;
        }

        public int ID { get; init; }
        
        [JsonIgnore] public Shape OutputShape { get; private set; }

        public Layer Construct()
        {
            var fork = new Fork();
            Forks[ID] = fork;
            return fork;
        }

        public Shape Initialize(Shape inputShape)
        {
            OutputShape = inputShape;
            return inputShape;
        }
    }
}
