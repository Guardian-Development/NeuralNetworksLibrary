using System;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class SynapseAssertor : Assertor<Synapse>
    {
        public int InputNeuronId { get; }
        public int OutputNeuronId { get; }

        public SynapseAssertor(Synapse expectedItem, int inputNeuronId, int outputNeuronId)
            : base(expectedItem)
        {
            InputNeuronId = inputNeuronId;
            OutputNeuronId = outputNeuronId;
        }

        public override void Assert(Synapse actualItem)
        {
            throw new NotImplementedException();
        }
    }

    public static class SynapseAssertorExtensions
    {
        public static SynapseAssertor ToAssertor(this Synapse expectedSynapse, int inputNeuronId, int outputNeuronId)
            => new SynapseAssertor(expectedSynapse, inputNeuronId, outputNeuronId);
    }
}