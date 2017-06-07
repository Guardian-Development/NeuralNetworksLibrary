using System;
using System.Collections.Generic;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class NeuralNetworkAssertorBuilder
    {
        private readonly List<(int id, Neuron neuron)> expectedNeurons = new List<(int id, Neuron neuron)>();
        private List<Synapse> expectedSynapses;

        public NeuralNetworkAssertorBuilder ExpectedNeurons(params (int id, Action<NeuronBuilder> builder)[] actions)
        {
            foreach (var action in actions)
            {
                var builder = new NeuronBuilder();
                action.builder.Invoke(builder);
                var neuron = builder.Build();
                expectedNeurons.Add((action.id, neuron));
            }
            return this;
        }

        public NeuralNetworkAssertorBuilder ExpectedSynapses(Action<SynapseBuilder> actions)
        {
            var builder = new SynapseBuilder();
            actions.Invoke(builder);
            expectedSynapses = builder.BuildWithoutConnetingNeurons(expectedNeurons);
            return this;
        }

        public NeuralNetworkAssertor Build() => NeuralNetworkAssertor.For(expectedNeurons, expectedSynapses);
    }
}