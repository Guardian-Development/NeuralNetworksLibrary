using System;
using NeuralNetworks.Library;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Library.Training.BackPropagation;
using NeuralNetworks.Tests.Support;
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
            throw new NotImplementedException();
        }

        public static BackPropagationSynapseWeightUpdateTester Create(double learningRate, double momentum)
            => new BackPropagationSynapseWeightUpdateTester(learningRate, momentum); 
    }
}