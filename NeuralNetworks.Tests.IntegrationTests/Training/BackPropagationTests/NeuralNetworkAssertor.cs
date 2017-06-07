using System;
using System.Collections.Generic;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class NeuralNetworkAssertor
    {
        private readonly List<(int id, Neuron neuron)> expectedNeurons;
        private readonly List<Synapse> expectedSynapses;

        private NeuralNetworkAssertor(List<(int id, Neuron neuron)> expectedNeurons, List<Synapse> expectedSynapses)
        {
            this.expectedNeurons = expectedNeurons;
            this.expectedSynapses = expectedSynapses;
        }

        public void Assert(
            List<(int id, Neuron neuron)> targetNeuralNetworkNeurons,
            List<Synapse> targetNeuralNetworkSynapses)
            => throw new NotImplementedException();

        public static NeuralNetworkAssertor For(
            List<(int id, Neuron neuron)> expectedNeurons,
            List<Synapse> expectedSynapses) => new NeuralNetworkAssertor(expectedNeurons, expectedSynapses);
    }
}