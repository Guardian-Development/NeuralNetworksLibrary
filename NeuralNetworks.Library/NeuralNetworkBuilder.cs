using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Components.Layers;
using NeuralNetworks.Library.Validation;

namespace NeuralNetworks.Library
{
    public sealed class NeuralNetworkBuilder
    {
        private InputLayer inputLayer;
        private readonly IList<HiddenLayer> hiddenLayers = new List<HiddenLayer>();
        private OutputLayer outputLayer;
        private double RandomDouble => randomNumberGenerator.NextDouble();
        private readonly Random randomNumberGenerator;

        public NeuralNetworkBuilder(Random randomNumberGenerator)
        {
            this.randomNumberGenerator = randomNumberGenerator;
        }

        public NeuralNetworkBuilder WithInputLayer(int neuronCount, ActivationType activationType)
        {
            inputLayer = InputLayer.For(neuronCount, activationType);
            return this;
        }

        public NeuralNetworkBuilder WithHiddenLayer(int neuronCount, ActivationType activationType)
        {
            var hiddenLayer = HiddenLayer.For(neuronCount, activationType);
            var previousLayer = GetPreviousLayer();

            hiddenLayers.Add(hiddenLayer);
            CreateNeuronConnectionsToPreviousLayer(previousLayer, hiddenLayer);
            previousLayer.NextLayer = hiddenLayer;

            return this;
        }

        public NeuralNetworkBuilder WithOutputLayer(int neuronCount, ActivationType activationType)
        {
            var previousLayer = GetPreviousLayer();

            outputLayer = OutputLayer.For(neuronCount, activationType);
            CreateNeuronConnectionsToPreviousLayer(previousLayer, outputLayer);
            previousLayer.NextLayer = outputLayer;

            return this;
        }

        public NeuralNetwork Build()
        {
            ValidateSpecifiedConfiguration();
            ConnectConfiguredLayers();
            return BuildNetwork();
        }

        private void ValidateSpecifiedConfiguration()
        {
            NullableValidators.ValidateNotNull(inputLayer);
            NullableValidators.ValidateNotNull(outputLayer);
        }

        private void ConnectConfiguredLayers()
        {
            ConnectInputLayerToNextLayer();
            ConnectHiddenLayersToNextLayer();
        }

        private NeuralNetwork BuildNetwork()
        {
            return new NeuralNetwork()
                .AddInputLayer(inputLayer)
                .AddHiddenLayers(hiddenLayers)
                .AddOutputLayer(outputLayer);
        }

        private void ConnectInputLayerToNextLayer()
        {
            inputLayer.NextLayer = hiddenLayers.Any() ? (Layer)hiddenLayers.First() : outputLayer;
        }

        private void ConnectHiddenLayersToNextLayer()
        {
            for (var i = 0; i < hiddenLayers.Count - 1; i++)
            {
                hiddenLayers[i].NextLayer = hiddenLayers[i + 1];
            }
            hiddenLayers.Last().NextLayer = outputLayer;
        }

        private Layer GetPreviousLayer()
        {
            return hiddenLayers.Any() ? (Layer)hiddenLayers.Last() : inputLayer;
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
    }
}
