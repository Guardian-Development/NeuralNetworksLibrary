
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
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0.78).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorRateForOutputLayerNeuron(
                    neuronId : 2, 
                    expectedOutput: 0.91,
                    errorRate : 0.04835);

             ErrorRateTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0.12333).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorRateForOutputLayerNeuron(
                    neuronId : 2, 
                    expectedOutput: 0.14,
                    errorRate : -0.09733);

             ErrorRateTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0.55552).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorRateForOutputLayerNeuron(
                    neuronId : 2, 
                    expectedOutput: 0.72,
                    errorRate : 0.01959);
        }

        [Fact]
        public void CanComputeErrorRateCorrectlyOutputLayerNeuronWhenErrorRateIsZero()
        {
            ErrorRateTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0.55552).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorRateForOutputLayerNeuron(
                    neuronId : 2, 
                    expectedOutput: 0.63542,
                    errorRate : 0);
        }
    }
}