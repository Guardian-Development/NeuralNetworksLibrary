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
        private readonly IList<Layer> hiddenLayers = new List<Layer>();
        private Layer outputLayer;

        private readonly IProvideRandomNumberGeneration randomNumberGenerator;

        public NeuralNetworkBuilder(IProvideRandomNumberGeneration randomNumberGenerator)
        {
            this.randomNumberGenerator = randomNumberGenerator;
        }

        public NeuralNetworkBuilder WithInputLayer(int neuronCount, ActivationType activationType)
        {
            var neurons =
                Enumerable.Repeat(
                    Neuron.For(activationType, randomNumberGenerator.GetNextRandomNumber()),
                    neuronCount);

            inputLayer = Layer.For(neurons);
            return this;
        }

        public NeuralNetworkBuilder WithHiddenLayer(int neuronCount, ActivationType activationType)
        {
            var previousLayer = GetPreviousLayer();
            var neurons =
                Enumerable.Repeat(
                    Neuron.For(
                        activationType,
                        randomNumberGenerator,
                        randomNumberGenerator.GetNextRandomNumber(),
                        previousLayer.Neurons), neuronCount); 

            hiddenLayers.Add(Layer.For(neurons));
            return this;
        }

        public NeuralNetworkBuilder WithOutputLayer(int neuronCount, ActivationType activationType)
        {
            var previousLayer = GetPreviousLayer();
            var neurons =
                Enumerable.Repeat(
                    Neuron.For(
                        activationType,
                        randomNumberGenerator,
                        randomNumberGenerator.GetNextRandomNumber(),
                        previousLayer.Neurons), neuronCount);

            outputLayer = Layer.For(neurons);

            return this;
        }

        private Layer GetPreviousLayer() =>
            hiddenLayers.Any() ? hiddenLayers.Last() : inputLayer;

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
                .AddHiddenLayers(hiddenLayers)
                .AddOutputLayer(outputLayer);
        }
    }
}
