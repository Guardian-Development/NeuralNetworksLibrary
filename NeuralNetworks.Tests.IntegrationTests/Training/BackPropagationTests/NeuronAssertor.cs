using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class NeuronAssertor : Assertor<Neuron>
    {
        private readonly int neuronId;
        private readonly List<SynapseAssertor> synapseAssertors;

        public NeuronAssertor(Neuron expectedItem, int neuronId, List<SynapseAssertor> synapseAssertors)
            : base(expectedItem)
        {
            this.neuronId = neuronId;
            this.synapseAssertors = synapseAssertors;
        }

        public override void Assert(Neuron actualItem)
        {
            actualItem.InputSynapses.ForEach(synapse => synapseAssertors
                .First(assertor => assertor.InputNeuronId == neuronId).Assert(synapse));

            actualItem.OutputSynapses.ForEach(synapse => synapseAssertors
                .First(assertor => assertor.OutputNeuronId == neuronId).Assert(synapse));

            //TODO: Assert all other features here. 

            throw new NotImplementedException();
        }
    }

    public static class NeuronAssertorExtensions
    {
        public static NeuronAssertor ToAssertor(this (int id, Neuron expectedNeuron) neuronWithId,
            List<SynapseAssertor> synapseAssertors)
            => new NeuronAssertor(neuronWithId.expectedNeuron, neuronWithId.id, synapseAssertors);
    }
}