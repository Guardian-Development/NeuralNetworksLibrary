using System;
using System.Linq;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Library.Training.BackPropagation;
using NeuralNetworks.Tests.Support;
using NeuralNetworks.Tests.Support.Assertors;
using NeuralNetworks.Tests.Support.Builders;
using NeuralNetworks.Tests.Support.Helpers;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests.SynapseWeightUpdateTests
{
    public sealed class BackPropagationSynapseWeightUpdateTester : NeuralNetworkTester<BackPropagationSynapseWeightUpdateTester>
    {
        private readonly SynapseWeightCalculator synapseWeightCalculator; 

        private BackPropagationSynapseWeightUpdateTester(double learningRate, double momentum)
        {
            synapseWeightCalculator = SynapseWeightCalculator.For(learningRate, momentum); 
        }

        public void UpdateSynapseExpectingWeight(
            int synapseInputNeuronId, 
            int synapseOutputNeuronId, 
            double expectedWeightDelta,
            double expectedWeight)
        {
            var neuron = FindNeuronWithId(synapseOutputNeuronId);
            var synapse = FindSynapseFor(neuron, synapseInputNeuronId); 
            synapseWeightCalculator.CalculateAndUpdateInputSynapseWeights(neuron); 

            var synapseAssertor = new SynapseAssertor.Builder()
                    .InputNeuronId(synapseInputNeuronId)
                    .OutputNeuronId(synapseOutputNeuronId)
                    .Weight(expectedWeight)
                    .WeightDelta(
                        expectedWeightDelta, 
                        precisionToAssertTo: targetNeuralNetwork.Context.SynapseWeightDecimalPlaces)
                    .Build(); 
            
            synapseAssertor.Assert(synapse);
        }

        private Synapse FindSynapseFor(Neuron neuron, int synapseInputNeuronId)
        { 
            var synapse = neuron.InputSynapses.First(s => s.InputNeuron.Id == synapseInputNeuronId);
            Assert.NotNull(synapse); 
            return synapse; 
        }

        public static BackPropagationSynapseWeightUpdateTester Create(double learningRate, double momentum)
            => new BackPropagationSynapseWeightUpdateTester(learningRate, momentum); 
    }
}