using System;
using System.Collections.Generic;
using NeuralNetworks.Library.Components.Layers;
using NeuralNetworks.Library.NetworkInitialisation;

namespace NeuralNetworks.Library
{
    public sealed class NeuralNetwork
    {
        public Layer InputLayer { get; private set; }
        public List<Layer> HiddenLayers { get; private set; }
        public Layer OutputLayer { get; private set; }

        public NeuralNetwork()
        {}

        public NeuralNetwork AddInputLayer(Layer inputLayer)
        {
            InputLayer = inputLayer;
            return this; 
        }

        public NeuralNetwork AddHiddenLayers(List<Layer> hiddenLayers)
        {
            HiddenLayers = hiddenLayers;
            return this; 
        }

        public NeuralNetwork AddOutputLayer(Layer outputLayer)
        {
            OutputLayer = outputLayer; 
            return this;
        }

        public static NeuralNetworkBuilder For(IProvideRandomNumberGeneration randomNumberGenerater = null)
        {
            randomNumberGenerater = randomNumberGenerater ?? RandomNumberProvider.For(new Random(1));
            return new NeuralNetworkBuilder(randomNumberGenerater);
        }
    }
}
