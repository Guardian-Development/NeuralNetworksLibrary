using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class NeuralNetworkAssertor
    {
        private readonly List<(int id, NeuronAssertor neuron)> expectedNeuronsAssertors;

        private NeuralNetworkAssertor(List<(int id, NeuronAssertor neuron)> expectedNeuronsAssertors)
        {
            this.expectedNeuronsAssertors = expectedNeuronsAssertors;
        }

        public void Assert(
            List<(int id, Neuron neuron)> targetNeuralNetworkNeurons,
            List<Synapse> targetNeuralNetworkSynapses)
        {
            ListAssertionHelpers.AssertEqualLength(targetNeuralNetworkNeurons, expectedNeuronsAssertors);
            AssertNeurons(targetNeuralNetworkNeurons);
        }

        private void AssertNeurons(List<(int id, Neuron neuron)> targetNeuralNetworkNeurons)
        {
            targetNeuralNetworkNeurons.ForEach(targetNeuron =>
                expectedNeuronsAssertors.First(assertor => assertor.id == targetNeuron.id)
                    .neuron.Assert(targetNeuron.neuron));
        }

        public static NeuralNetworkAssertor For(List<(int id, NeuronAssertor neuron)> expectedNeuronsAssertors)
            => new NeuralNetworkAssertor(expectedNeuronsAssertors);
    }
}