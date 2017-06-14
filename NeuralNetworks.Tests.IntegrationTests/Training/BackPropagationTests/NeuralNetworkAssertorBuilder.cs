using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class NeuralNetworkAssertorBuilder
    {
        private readonly List<(int id, Neuron neuron)> expectedNeurons = new List<(int id, Neuron neuron)>();
        private List<SynapseAssertor> synapseAssertors;

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
            synapseAssertors = builder.BuildWithoutConnectingNeurons(expectedNeurons);
            return this;
        }

        public NeuralNetworkAssertor Build()
        {
            var neuronAssertors = expectedNeurons
                .Select(neuronWithId => ValueTuple.Create(neuronWithId.id, neuronWithId.ToAssertor(synapseAssertors)))
                .ToList();

            return NeuralNetworkAssertor.For(neuronAssertors, synapseAssertors);
        }
    }
}