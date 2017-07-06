using System;
using System.Collections.Generic;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.NetworkInitialisation;

namespace NeuralNetworks.Tests.Support.Builders
{
    public class ExplicitNeuralNetworkBuilder
    {
        private readonly IDictionary<int, Neuron> allNeuronsInNetwork = new Dictionary<int, Neuron>();

        private InputLayer inputLayer;
        private List<HiddenLayer> hiddenLayers = new List<HiddenLayer>();
        private OutputLayer outputLayer; 

		private readonly NeuralNetworkContext context;
        private readonly IProvideRandomNumberGeneration randomNumberGenerator;

        private ExplicitNeuralNetworkBuilder(
            NeuralNetworkContext context, 
            IProvideRandomNumberGeneration randomNumberGenerator)
        {
            this.randomNumberGenerator = randomNumberGenerator;
            this.context = context;
        }

        public ExplicitNeuralNetworkBuilder InputLayer(Action<InputlayerBuilder> action)
        {
            var builder = new InputlayerBuilder(allNeuronsInNetwork, context); 
            action.Invoke(builder);

            inputLayer = builder.Build();
            return this; 
        }

        public ExplicitNeuralNetworkBuilder HiddenLayer(Action<HiddenLayerBuilder> action)
        {
            var builder = new HiddenLayerBuilder(allNeuronsInNetwork, context);
			action.Invoke(builder);

            hiddenLayers.Add(builder.Build());
            return this; 
        }

        public ExplicitNeuralNetworkBuilder OutputLayer(Action<OutputLayerBuilder> action)
        {
            var builder = new OutputLayerBuilder(allNeuronsInNetwork, context);
			action.Invoke(builder);

            outputLayer = builder.Build();
			return this;
		}

        public ExplicitNeuralNetworkBuilder Synapses(params Action<SynapseBuilder>[] actions)
        {
            foreach(Action<SynapseBuilder> action in actions)
            {
                var synapseBuilder = new SynapseBuilder(context, allNeuronsInNetwork, randomNumberGenerator);
                action.Invoke(synapseBuilder);
                synapseBuilder.Build();
            }

            return this; 
        }

        public NeuralNetwork Build()
        {
            return new NeuralNetwork(context)
                .AddInputLayer(inputLayer)
                .AddHiddenLayers(hiddenLayers)
                .AddOutputLayer(outputLayer);
        }

        public static ExplicitNeuralNetworkBuilder CreateForTest(
            NeuralNetworkContext context,
            IProvideRandomNumberGeneration randomNumberGenerator) 
            => new ExplicitNeuralNetworkBuilder(context, randomNumberGenerator);
    }
}
