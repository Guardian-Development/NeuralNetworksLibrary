using System;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class SynapseAssertor : Assertor<Synapse>
    {
        public SynapseAssertor(Synapse expectedItem): base(expectedItem)
        {}

        public override void Assert(Synapse actualItem)
        {
            throw new NotImplementedException();
        }
    }

    public static class SynapseAssertorExtensions
    {
        public static SynapseAssertor ToAssertor(this Synapse expectedSynapse)
            => new SynapseAssertor(expectedSynapse);
    }
}
