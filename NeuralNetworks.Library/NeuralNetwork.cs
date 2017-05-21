using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Components.Layers;

namespace NeuralNetworks.Library
{
    public sealed class NeuralNetwork
    {
        private InputLayer inputLayer;
        private readonly IList<Layer> hiddenLayers = new List<Layer>();
        private OutputLayer outputLayer;

        private double RandomDouble => randomNumberGenerator.NextDouble(); 
        private readonly Random randomNumberGenerator;

        private NeuralNetwork(Random randomNumberGenerator)
        {
            this.randomNumberGenerator = randomNumberGenerator; 
        }

        public NeuralNetwork WithInputLayer(int neuronCount, ActivationType activationType)
        {
            inputLayer = InputLayer.For(neuronCount, activationType);
            return this;
        }

        public NeuralNetwork WithHiddenLayer(int neuronCount, ActivationType activationType)
        {
            var previousLayer = GetPreviousLayer();
            var hiddenLayer = HiddenLayer.For(neuronCount, activationType, previousLayer);

            hiddenLayers.Add(hiddenLayer);
            CreateNeuronConnectionsToPreviousLayer(previousLayer, hiddenLayer);

            return this;
        }

        public NeuralNetwork WithOutputLayer(int neuronCount, ActivationType activationType)
        {
            var previousLayer = GetPreviousLayer();

            outputLayer = OutputLayer.For(neuronCount, activationType, previousLayer);
            CreateNeuronConnectionsToPreviousLayer(previousLayer, outputLayer);

            return this;
        }

        public double[] MakePrediction(double[] inputs)
        {
            PopulateInputLayer(inputs);

            inputLayer
                .Concat(hiddenLayers)
                .Concat(outputLayer)
                .ToList()
                .ForEach(layer => layer.ActivateLayer());

            return outputLayer.GetPrediction(); 
        }

        private Layer GetPreviousLayer()
        {
            return hiddenLayers.Any() ? hiddenLayers.Last() : inputLayer;
        }

        private void CreateNeuronConnectionsToPreviousLayer(Layer previousLayer, Layer newLayer)
        {
            foreach (var newLayerNeuron in newLayer.Neurons)
            {
                AddNeuronConnection(previousLayer, newLayerNeuron);
            }
        }

        private void AddNeuronConnection(Layer previousLayer, Neuron newLayerNeuron)
        {
            foreach (var previousLayerNeuron in previousLayer.Neurons)
            {
                newLayerNeuron.AddInputConnection(Synapse.For(previousLayerNeuron, RandomDouble));
            }
        }

        private void PopulateInputLayer(double[] input)
        {
            PerformInputLayerConfigurationChecks(input.Length);
            for (var i = 0; i < inputLayer.Neurons.Length; i++)
            {
                inputLayer.Neurons[i].Output = input[i];
            }
        }

        private void PerformInputLayerConfigurationChecks(int inputSize)
        {
            if (inputLayer == null)
            {
                throw new InvalidOperationException(
                    $"You must specify an input layer before populating, call {nameof(WithInputLayer)} first.");
            }

            if (inputLayer.Neurons.Length != inputSize)
            {
                throw new ArgumentException($"{nameof(inputLayer)} Neurons count must be the same length as the {inputSize}");
            }
        }

        public static NeuralNetwork Create(Random randomNumberGenerator = null)
        {
            randomNumberGenerator = randomNumberGenerator ?? new Random(1);
            return new NeuralNetwork(randomNumberGenerator);
        }
    }
}
