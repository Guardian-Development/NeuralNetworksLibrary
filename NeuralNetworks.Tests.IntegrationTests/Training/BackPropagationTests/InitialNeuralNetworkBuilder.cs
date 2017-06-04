using System;
using System.Collections.Generic;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class InitialNeuralNetworkBuilder
    {
        private InputLayer inputLayer;
        private readonly List<HiddenLayer> hiddenLayers = new List<HiddenLayer>();
        private OutputLayer outputLayer;

        private List<Synapse> synapses;
        private readonly List<(int id, Neuron neuron)> allNeurons = new List<(int id, Neuron neuron)>();

        public InitialNeuralNetworkBuilder InputLayer(Action<InputLayerBuilder> actions)
        {
            var builder = new InputLayerBuilder();
            actions.Invoke(builder);
            allNeurons.AddRange(builder.AllNeurons);
            inputLayer = builder.Build();
            return this;
        }

        public InitialNeuralNetworkBuilder HiddenLayer(Action<HiddenLayerBuilder> actions)
        {
            var builder = new HiddenLayerBuilder();
            actions.Invoke(builder);
            allNeurons.AddRange(builder.AllNeurons);
            hiddenLayers.Add(builder.Build());
            return this;
        }

        public InitialNeuralNetworkBuilder OutputLayer(Action<OutputLayerBuilder> actions)
        {
            var builder = new OutputLayerBuilder();
            actions.Invoke(builder);
            allNeurons.AddRange(builder.AllNeurons);
            outputLayer = builder.Build();
            return this;
        }

        public InitialNeuralNetworkBuilder Synapses(Action<SynapseBuilder> actions)
        {
            var builder = new SynapseBuilder();
            actions.Invoke(builder);
            synapses = builder.Build(allNeurons);
            return this; 
        }

        public (NeuralNetwork network, List<(int id, Neuron neuron)> allNeurons, List<Synapse> allSynapses) Build()
        {
            //connect neurons together here. 
            throw new NotImplementedException();
        }
    }
}