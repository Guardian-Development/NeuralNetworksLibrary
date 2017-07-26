
using System;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests.ErrorRateCalculationsTests
{
    public sealed class BackPropagationErrorRateCalculationTests : NeuralNetworkTest
    {
        public Func<BackPropagationErrorRateTester> ErrorRateTester 
            => BackPropagationErrorRateTester.Create;
            
        private NeuralNetworkContext TestContext => 
            new NeuralNetworkContext(
                errorRateDecimalPlaces: 5, 
                outputDecimalPlaces: 5, 
                synapseWeightDecimalPlaces: 5); 

        [Fact]
        public void CanComputeErrorRateCorrectlyOutputLayerNeuron()
        {
             ErrorRateTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.15)))
                .CalculateErrorRateForOutputLayerNeuron(
                    neuronId : 2, 
                    expectedOutput: 0.4,
                    expectedErrorRate : 0.5);
        }
    }
}