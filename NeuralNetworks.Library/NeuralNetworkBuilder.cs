using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Library.Validation;

namespace NeuralNetworks.Library
{
    public sealed class NeuralNetworkBuilder
    {
        private InputLayer inputLayer;
        private readonly List<HiddenLayer> hiddenLayers = new List<HiddenLayer>();
        private OutputLayer outputLayer;

        private Layer PreviousLayer => hiddenLayers.Any() ? (Layer)hiddenLayers.Last() : inputLayer;

        private readonly NeuralNetworkContext context;
		private readonly IProvideRandomNumberGeneration randomNumberGenerator;

        public NeuralNetworkBuilder(NeuralNetworkContext context, IProvideRandomNumberGeneration randomNumberGenerator)
        {
            this.context = context;
            this.randomNumberGenerator = randomNumberGenerator;
        }

        public NeuralNetworkBuilder WithInputLayer(int neuronCount, ActivationType activationType, double biasOutput = 1)
        {
            var neurons = new List<Neuron>();

            for (var i = 0; i < neuronCount; i++)
            {
                neurons.Add(Neuron.For(context, activationType));
            }

            inputLayer = InputLayer.For(neurons, BiasNeuron.For(context, activationType, biasOutput));
            return this;
        }

        public NeuralNetworkBuilder WithHiddenLayer(int neuronCount, ActivationType activationType, double biasOutput = 1)
        {
            var neurons = new List<Neuron>();

            for (var i = 0; i < neuronCount; i++)
            {
                neurons.Add(Neuron.For(
                    context,
                    activationType,
                    randomNumberGenerator,
                    PreviousLayer.Neurons));
            }

            hiddenLayers.Add(HiddenLayer.For(neurons, BiasNeuron.For(context, activationType, biasOutput)));
            return this;
        }

        public NeuralNetworkBuilder WithOutputLayer(int neuronCount, ActivationType activationType)
        {
            var neurons = new List<Neuron>();

            for (var i = 0; i < neuronCount; i++)
            {
                neurons.Add(Neuron.For(
                    context,
                    activationType,
                    randomNumberGenerator,
                    PreviousLayer.Neurons));
            }

            outputLayer = OutputLayer.For(neurons);

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
            return new NeuralNetwork(context)
                .AddInputLayer(inputLayer)
                .AddHiddenLayers(hiddenLayers)
                .AddOutputLayer(outputLayer);
        }
    }
}