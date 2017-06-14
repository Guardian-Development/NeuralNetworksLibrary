using System;
using System.Collections.Generic;
using System.Linq;
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

        public List<Synapse> BuildConnectingNeurons(List<(int id, Neuron neuron)> neuronsWithId)
        {
            var connectedSynapses = new List<Synapse>();
            foreach (var synapseStructure in synapses)
            {
                var inputNeuron = FindNeuronById(neuronsWithId, synapseStructure.inputNeuronId);
                var outputNeuron = FindNeuronById(neuronsWithId, synapseStructure.outputNeuronId);

                var synapse = Synapse.For(inputNeuron, outputNeuron, PredictableRandomNumberGenerator.Create());
                synapse.Weight = synapseStructure.weight;

                inputNeuron.OutputSynapses.Add(synapse);
                outputNeuron.InputSynapses.Add(synapse);
                connectedSynapses.Add(synapse);
            }
            return connectedSynapses;
        }

        public List<SynapseAssertor> BuildWithoutConnectingNeurons(List<(int id, Neuron neuron)> neuronsWithId)
        {
            var nonConnectedSynapses = new List<SynapseAssertor>();
            foreach (var synapseStructure in synapses)
            {
                var inputNeuron = FindNeuronById(neuronsWithId, synapseStructure.inputNeuronId);
                var outputNeuron = FindNeuronById(neuronsWithId, synapseStructure.outputNeuronId);

                var synapse = Synapse.For(inputNeuron, outputNeuron, PredictableRandomNumberGenerator.Create());
                synapse.Weight = synapseStructure.weight;
                nonConnectedSynapses.Add(synapse.ToAssertor(
                    synapseStructure.inputNeuronId,
                    synapseStructure.outputNeuronId));
            }
            return nonConnectedSynapses;
        }

        private static Neuron FindNeuronById(IEnumerable<(int id, Neuron neuron)> neurons, int id)
            => neurons.Where(neuron => neuron.id == id)
                .Select(neuron => neuron.neuron)
                .First();
    }
}