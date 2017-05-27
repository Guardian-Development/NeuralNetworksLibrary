using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components.Layers;
using NeuralNetworks.Library.NetworkInitialisation;

namespace NeuralNetworks.Library
{
    public sealed class NeuralNetwork
    {
        public Layer InputLayer { get; private set; }
        public List<Layer> HiddenLayers { get; private set; }
        public Layer OutputLayer { get; private set; }

        internal NeuralNetwork()
        {}

        internal NeuralNetwork AddInputLayer(Layer inputLayer)
        {
            InputLayer = inputLayer;
            return this; 
        }

        internal NeuralNetwork AddHiddenLayers(List<Layer> hiddenLayers)
        {
            HiddenLayers = hiddenLayers;
            return this; 
        }

        internal NeuralNetwork AddOutputLayer(Layer outputLayer)
        {
            OutputLayer = outputLayer; 
            return this;
        }

        public double[] PredictionFor(params double[] inputs)
        {
            ValidateInputs(inputs.Length);

            var i = 0;
            InputLayer.Neurons.ForEach(a => a.Value = inputs[i++]);
            HiddenLayers.ApplyInReverse(layer => layer.Neurons.ForEach(a => a.CalculateOutput()));
            return OutputLayer.Neurons.Select(a => a.CalculateOutput()).ToArray();
        }

        private void ValidateInputs(int inputsLength)
        {
            if (inputsLength != InputLayer.Neurons.Count)
            {
                throw new ArgumentException(
                    "Input length must be the same length as the Input Layer Neurons");
            }
        }

        public static NeuralNetworkBuilder For(IProvideRandomNumberGeneration randomNumberGenerater = null)
        {
            randomNumberGenerater = randomNumberGenerater ?? RandomNumberProvider.For(new Random(1));
            return new NeuralNetworkBuilder(randomNumberGenerater);
        }
    }
}
