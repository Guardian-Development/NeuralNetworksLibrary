using System;
using System.Net;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class NeuronAssertor : Assertor<Neuron>
    {
        private readonly int neuronId;

        public NeuronAssertor(Neuron expectedItem, int neuronId) : base(expectedItem)
        {
            this.neuronId = neuronId; 
        }

        public override void Assert(Neuron actualItem)
        {
            throw new NotImplementedException();
        }
    }

    public static class NeuronAssertorExtensions
    {
        public static NeuronAssertor ToAssertor(this (int id, Neuron expectedNeuron) neuronWithId) 
            => new NeuronAssertor(neuronWithId.expectedNeuron, neuronWithId.id);
    }
}