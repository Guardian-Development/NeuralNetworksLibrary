using System;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.NetworkPredictions
{
    public sealed class NeuralNetworkFeedForwardTests : NeuralNetworkTest
    {
        public Func<NeuralNetworkPredictionsTester> FeedForwardTester 
            => NeuralNetworkPredictionsTester.Create;
        private NeuralNetworkContext TestContext => 
            new NeuralNetworkContext(
                errorRateDecimalPlaces: 5, 
                outputDecimalPlaces: 5, 
                synapseWeightDecimalPlaces: 5); 

        [Fact]
        public void CanFeedForwardOutputCorrectlyForSingleInputNeuronNoHiddenLayerSingleOutputNeuron()
        {
             FeedForwardTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.15)))
                .InputAndExpectOutput(
                    inputs: new [] { 0.1 },
                    expectedOutput: new [] { 0.5 }); 
        }
    }
}