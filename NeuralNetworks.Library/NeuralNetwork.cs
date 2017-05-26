using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using NeuralNetworks.Library.Components.Layers;
using NeuralNetworks.Library.NetworkInitialisation;

namespace NeuralNetworks.Library
{
    public sealed class NeuralNetwork
    {
        public Layer InputLayer { get; private set; }
        public IReadOnlyList<Layer> HiddenLayers { get; private set; }
        public Layer OutputLayer { get; private set; }

        internal NeuralNetwork()
        {}

        internal NeuralNetwork AddInputLayer(Layer inputLayer)
        {
            InputLayer = inputLayer;
            return this; 
        }

        internal NeuralNetwork AddHiddenLayers(IList<Layer> hiddenLayers)
        {
            HiddenLayers = new ReadOnlyCollection<Layer>(hiddenLayers);
            return this; 
        }

        internal NeuralNetwork AddOutputLayer(Layer outputLayer)
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
