using System;
using System.Collections.Generic;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class SynapseBuilder
    {
        private readonly List<(int inputNeuronId, int outputNeuronId, double weight)> synapses =
            new List<(int inputNeuronId, int outputNeuronId, double weight)>();

        public SynapseBuilder SynapseBetween(int inputNeuronId, int outputNeuronId, double weight)
        {
            synapses.Add(ValueTuple.Create(inputNeuronId, outputNeuronId, weight));
            return this; 
        }

        public List<Synapse> Build(List<(int id, Neuron neuron)> neuronsWithId)
            => throw new NotImplementedException();
    }
}