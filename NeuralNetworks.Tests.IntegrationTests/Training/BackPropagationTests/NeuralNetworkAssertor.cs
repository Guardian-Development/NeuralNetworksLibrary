using System;
using System.Collections.Generic;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class NeuralNetworkAssertor
    {
        private readonly List<(int id, NeuronAssertor neuron)> expectedNeuronsAssertors;
        private readonly List<SynapseAssertor> expectedSynapsesAssertors;

        private NeuralNetworkAssertor(
            List<(int id, NeuronAssertor neuron)> expectedNeuronsAssertors, 
            List<SynapseAssertor> expectedSynapsesAssertors)
        {
            this.expectedNeuronsAssertors = expectedNeuronsAssertors;
            this.expectedSynapsesAssertors = expectedSynapsesAssertors;
        }

        public void Assert(
            List<(int id, Neuron neuron)> targetNeuralNetworkNeurons,
            List<Synapse> targetNeuralNetworkSynapses)
            => throw new NotImplementedException();

        public static NeuralNetworkAssertor For(
            List<(int id, NeuronAssertor neuron)> expectedNeuronsAssertors,
            List<SynapseAssertor> expectedSynapsesAssertors) 
            => new NeuralNetworkAssertor(expectedNeuronsAssertors, expectedSynapsesAssertors);
    }
}