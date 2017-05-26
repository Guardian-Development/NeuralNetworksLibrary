using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Components.Layers;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Library.Validation;

namespace NeuralNetworks.Library
{
    public sealed class NeuralNetworkBuilder
    {
        private Layer inputLayer;
        private Layer hiddenLayer;
        private Layer outputLayer;

        private readonly IProvideRandomNumberGeneration randomNumberGenerator;

        public NeuralNetworkBuilder(IProvideRandomNumberGeneration randomNumberGenerator)
        {
            this.randomNumberGenerator = randomNumberGenerator;
        }

        public NeuralNetworkBuilder WithInputLayer(int neuronCount, ActivationType activationType)
        {
            var neurons = new List<Neuron>();

            for (var i = 0; i < neuronCount; i++)
            {
                neurons.Add(Neuron.For(activationType, randomNumberGenerator.GetNextRandomNumber()));
            }

            inputLayer = Layer.For(neurons);
            return this;
        }

        public NeuralNetworkBuilder WithHiddenLayer(int neuronCount, ActivationType activationType)
        {
            var neurons = new List<Neuron>();

            for (var i = 0; i < neuronCount; i++)
            {
                neurons.Add(Neuron.For(
                    activationType,
                    randomNumberGenerator,
                    randomNumberGenerator.GetNextRandomNumber(),
                    inputLayer.Neurons));
            }

            hiddenLayer = Layer.For(neurons);
            return this;
        }

        public NeuralNetworkBuilder WithOutputLayer(int neuronCount, ActivationType activationType)
        {
            var neurons = new List<Neuron>();

            for (var i = 0; i < neuronCount; i++)
            {
                neurons.Add(Neuron.For(
                    activationType,
                    randomNumberGenerator,
                    randomNumberGenerator.GetNextRandomNumber(),
                    hiddenLayer.Neurons));
            }

            outputLayer = Layer.For(neurons);

            return this;
        }

        public NeuralNetwork Build()
        {
            ValidateSpecifiedConfiguration();
            return BuildNetwork();
        }

        private void ValidateSpecifiedConfiguration()
        {
            NullableValidators.ValidateNotNull(inputLayer);
            NullableValidators.ValidateNotNull(outputLayer);
        }

        private NeuralNetwork BuildNetwork()
        {
            return new NeuralNetwork()
                .AddInputLayer(inputLayer)
                .AddHiddenLayer(hiddenLayer)
                .AddOutputLayer(outputLayer);
        }
    }
}