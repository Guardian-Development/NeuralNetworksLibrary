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
        protected readonly IDictionary<int, Neuron> AllNeuronsInNetwork;
        protected readonly NeuralNetworkContext Context;

        protected IList<Neuron> NeuronsInLayer = new List<Neuron>(); 

        public LayerBuilder(IDictionary<int, Neuron> allNeuronsInNetwork, NeuralNetworkContext context)
        {
            AllNeuronsInNetwork = allNeuronsInNetwork;
			Context = context;
        }

        public TLayer Neurons(params Action<NeuronBuilder>[] actions)
        {
            foreach(var action in actions)
            {
                var neuronBuilder = new NeuronBuilder(Context);
                action.Invoke(neuronBuilder);

                var neuron = neuronBuilder.Build(); 

                NeuronsInLayer.Add(neuron);
                AllNeuronsInNetwork.Add(neuron.Id, neuron);
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
            var neuronBuilder = new BiasNeuronBuilder(Context); 
            action.Invoke(neuronBuilder);
            biasNeuron = neuronBuilder.Build();

            NeuronsInLayer.Add(biasNeuron);
            AllNeuronsInNetwork.Add(biasNeuron.Id, biasNeuron);

            return this; 
        }

        public InputLayer Build()
        {
            return biasNeuron == null ? 
                InputLayer.For(NeuronsInLayer.ToList()) : 
                InputLayer.For(NeuronsInLayer.ToList(), biasNeuron);
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
			var neuronBuilder = new BiasNeuronBuilder(Context);
			action.Invoke(neuronBuilder);
			biasNeuron = neuronBuilder.Build();

            NeuronsInLayer.Add(biasNeuron);
            AllNeuronsInNetwork.Add(biasNeuron.Id, biasNeuron);

            return this;
		}

        public HiddenLayer Build()
        {
            return biasNeuron == null ? 
                HiddenLayer.For(NeuronsInLayer.ToList()) : 
                HiddenLayer.For(NeuronsInLayer.ToList(), biasNeuron);
        }
    }

    public sealed class OutputLayerBuilder : LayerBuilder<OutputLayerBuilder>
    {
        public OutputLayerBuilder(IDictionary<int, Neuron> allNeuronsInNetwork, NeuralNetworkContext context) 
            : base(allNeuronsInNetwork, context)
        {}

        public OutputLayer Build()
            => OutputLayer.For(NeuronsInLayer.ToList()); 
    }
}
