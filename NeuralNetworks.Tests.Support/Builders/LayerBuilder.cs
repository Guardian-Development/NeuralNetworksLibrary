using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.Support.Builders
{
    public class LayerBuilder<TLayer>
        where TLayer : LayerBuilder<TLayer>
    {
        protected readonly IDictionary<int, Neuron> allNeuronsInNetwork;
        protected readonly NeuralNetworkContext context;

        protected IList<Neuron> neuronsInLayer = new List<Neuron>(); 

        public LayerBuilder(IDictionary<int, Neuron> allNeuronsInNetwork, NeuralNetworkContext context)
        {
            this.allNeuronsInNetwork = allNeuronsInNetwork;
			this.context = context;
        }

        public TLayer Neurons(params Action<NeuronBuilder>[] actions)
        {
            foreach(Action<NeuronBuilder> action in actions)
            {
                var neuronBuilder = new NeuronBuilder(context);
                action.Invoke(neuronBuilder);

                var neuron = neuronBuilder.Build(); 

                neuronsInLayer.Add(neuron);
                allNeuronsInNetwork.Add(neuron.Id, neuron);
            }

            return (TLayer)this; 
        }
    }

    public sealed class InputlayerBuilder : LayerBuilder<InputlayerBuilder>
    {
        private BiasNeuron biasNeuron; 

        public InputlayerBuilder(IDictionary<int, Neuron> allNeuronsInNetwork, NeuralNetworkContext context) 
            : base(allNeuronsInNetwork, context)
        {}

        public InputlayerBuilder Bias(Action<BiasNeuronBuilder> action)
        {
            var neuronBuilder = new BiasNeuronBuilder(context); 
            action.Invoke(neuronBuilder);
            biasNeuron = neuronBuilder.Build();

            return this; 
        }

        public InputLayer Build()
        {
            return biasNeuron == null ? 
                InputLayer.For(neuronsInLayer.ToList()) : 
                InputLayer.For(neuronsInLayer.ToList(), biasNeuron);
        }
    }

    public sealed class HiddenLayerBuilder : LayerBuilder<HiddenLayerBuilder>
    {
        private BiasNeuron biasNeuron; 

        public HiddenLayerBuilder(IDictionary<int, Neuron> allNeuronsInNetwork, NeuralNetworkContext context)
            : base(allNeuronsInNetwork, context)
        {}

        public HiddenLayerBuilder Bias(Action<BiasNeuronBuilder> action)
        {
			var neuronBuilder = new BiasNeuronBuilder(context);
			action.Invoke(neuronBuilder);
			biasNeuron = neuronBuilder.Build();

			return this;
		}

        public HiddenLayer Build()
        {
            return biasNeuron == null ? 
                HiddenLayer.For(neuronsInLayer.ToList()) : 
                HiddenLayer.For(neuronsInLayer.ToList(), biasNeuron);
        }
    }

    public sealed class OutputLayerBuilder : LayerBuilder<OutputLayerBuilder>
    {
        public OutputLayerBuilder(IDictionary<int, Neuron> allNeuronsInNetwork, NeuralNetworkContext context) 
            : base(allNeuronsInNetwork, context)
        {}

        public OutputLayer Build()
            => OutputLayer.For(neuronsInLayer.ToList()); 
    }
}
