using System;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests.SynapseWeightUpdateTests
{
    public sealed class BackPropagationSynapseWeightUpdateTests : NeuralNetworkTest
    {
        public BackPropagationSynapseWeightUpdateTester SynapseWeightUpdateTester(double learningRate, double momentum)
            => BackPropagationSynapseWeightUpdateTester.Create(learningRate, momentum);
            
        private NeuralNetworkContext TestContext => 
            new NeuralNetworkContext(
                errorRateDecimalPlaces: 5, 
                outputDecimalPlaces: 5, 
                synapseWeightDecimalPlaces: 5); 

        [Fact(Skip="Not Implemented")]
        public void CanUpdateSynapseWeightCorrectlyInputLayerNeuronToHiddenLayerNeuron()
        {
             SynapseWeightUpdateTester(learningRate: 1, momentum: 1)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0.78).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 1, 
                    synapseOutputNeuronId: 2, 
                    expectedWeight: 0.11);
        }

        [Fact(Skip="Not Implemented")]
        public void CanUpdateSynapseWeightCorrectlyInputLayerNeuronToOutputLayerNeuron()
        {
            
        }

        [Fact(Skip="Not Implemented")]
        public void CanUpdateSynapseWeightCorrectlyHiddenLayerNeuronToHiddenLayerNeuron()
        {
            
        }

        [Fact(Skip="Not Implemented")]
        public void CanUpdateSynapseWeightCorrectlyHiddenLayerNeuronToOutputLayerNeuron()
        {
            
        }
    }
}