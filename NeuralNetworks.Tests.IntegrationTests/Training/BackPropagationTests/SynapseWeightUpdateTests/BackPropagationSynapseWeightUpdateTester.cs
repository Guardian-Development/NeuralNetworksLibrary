using System;
using System.Linq;
using NeuralNetworks.Library;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Library.Training.BackPropagation;
using NeuralNetworks.Tests.Support;
using NeuralNetworks.Tests.Support.Assertors;
using NeuralNetworks.Tests.Support.Builders;

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
            double expectedWeight)
        {
            var inputNeuron = FindNeuronWithId(synapseInputNeuronId); 
            var synapse = inputNeuron.OutputSynapses.First(s => s.OutputNeuron.Id == synapseOutputNeuronId);
            var synapseAssertor = new SynapseAssertor.Builder()
                    .InputNeuronId(synapseInputNeuronId)
                    .OutputNeuronId(synapseOutputNeuronId)
                    .Weight(expectedWeight)
                    .Build(); 
            
            synapseAssertor.Assert(synapse); 
        }

        public static BackPropagationSynapseWeightUpdateTester Create(double learningRate, double momentum)
            => new BackPropagationSynapseWeightUpdateTester(learningRate, momentum); 
    }
}