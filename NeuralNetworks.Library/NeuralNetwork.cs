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
        internal InputLayer InputLayer;
        internal readonly IList<HiddenLayer> HiddenLayers = new List<HiddenLayer>();
        internal OutputLayer OutputLayer;

        private double RandomDouble => randomNumberGenerator.NextDouble(); 
        private readonly Random randomNumberGenerator;

        private NeuralNetwork(Random randomNumberGenerator)
        {
            this.randomNumberGenerator = randomNumberGenerator; 
        }

        public NeuralNetwork WithInputLayer(int neuronCount, ActivationType activationType)
        {
            InputLayer = InputLayer.For(neuronCount, activationType);
            return this;
        }

        public NeuralNetwork WithHiddenLayer(int neuronCount, ActivationType activationType)
        {
            var hiddenLayer = HiddenLayer.For(neuronCount, activationType);
            var previousLayer = GetPreviousLayer();

            HiddenLayers.Add(hiddenLayer);
            CreateNeuronConnectionsToPreviousLayer(previousLayer, hiddenLayer);
            previousLayer.NextLayer = hiddenLayer;

            return this;
        }

        public NeuralNetwork WithOutputLayer(int neuronCount, ActivationType activationType)
        {
            var previousLayer = GetPreviousLayer();

            OutputLayer = OutputLayer.For(neuronCount, activationType);
            CreateNeuronConnectionsToPreviousLayer(previousLayer, OutputLayer);
            previousLayer.NextLayer = OutputLayer; 

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

        private Layer GetPreviousLayer()
        {
            return HiddenLayers.Any() ? (Layer) HiddenLayers.Last() : InputLayer;
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
                    $"You must specify an input layer before populating, call {nameof(WithInputLayer)} first.");
            }

            if (InputLayer.Neurons.Length != inputSize)
            {
                throw new ArgumentException($"{nameof(InputLayer)} Neurons count must be the same length as the {inputSize}");
            }
        }

        public static NeuralNetwork Create(Random randomNumberGenerator = null)
        {
            randomNumberGenerator = randomNumberGenerator ?? new Random(1);
            return new NeuralNetwork(randomNumberGenerator);
        }
    }
}
