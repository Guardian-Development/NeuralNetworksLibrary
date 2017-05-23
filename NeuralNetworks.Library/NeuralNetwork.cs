using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using NeuralNetworks.Library.Components.Layers;

namespace NeuralNetworks.Library
{
    public sealed class NeuralNetwork
    {
        internal InputLayer InputLayer { get; private set; }
        internal IReadOnlyList<HiddenLayer> HiddenLayers { get; private set; }
        internal OutputLayer OutputLayer { get; private set; }

        internal NeuralNetwork()
        {}

        internal NeuralNetwork AddInputLayer(InputLayer inputLayer)
        {
            InputLayer = inputLayer;
            return this; 
        }

        internal NeuralNetwork AddHiddenLayers(IList<HiddenLayer> hiddenLayers)
        {
            HiddenLayers = new ReadOnlyCollection<HiddenLayer>(hiddenLayers);
            return this; 
        }

        internal NeuralNetwork AddOutputLayer(OutputLayer outputLayer)
        {
            OutputLayer = outputLayer; 
            return this;
        }

        public double[] MakePrediction(double[] inputs)
        {
            PopulateInputLayer(inputs);

            InputLayer
                .Concat(HiddenLayers)
                .Concat(OutputLayer)
                .ToList()
                .ForEach(layer => layer.ActivateLayer());

            return OutputLayer.GetPrediction(); 
        }

        private void PopulateInputLayer(double[] input)
        {
            PerformInputLayerConfigurationChecks(input.Length);
            for (var i = 0; i < InputLayer.Neurons.Length; i++)
            {
                InputLayer.Neurons[i].Output = input[i];
            }
        }

        private void PerformInputLayerConfigurationChecks(int inputSize)
        {
            if (InputLayer == null)
            {
                throw new InvalidOperationException(
                    $"You must specify an input layer before populating, call {nameof(AddInputLayer)} first.");
            }

            if (InputLayer.Neurons.Length != inputSize)
            {
                throw new ArgumentException($"{nameof(InputLayer)} Neurons count must be the same length as the {inputSize}");
            }
        }

        public static NeuralNetworkBuilder For(Random randomNumberGenerator = null)
        {
            randomNumberGenerator = randomNumberGenerator ?? new Random(1);
            return new NeuralNetworkBuilder(randomNumberGenerator);
        }
    }
}
