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

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyInputLayerNeuronToHiddenLayerNeuron()
        {
             SynapseWeightUpdateTester(learningRate: 1, momentum: 0)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0.15).Output(0.78).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0.11).Output(0.91).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(3).ErrorRate(0.98).Output(0.77).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.45),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.11)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 1,
                    synapseOutputNeuronId: 2,
                    expectedWeightDelta : 0.0858,
                    expectedWeight: 0.5358);
        }

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyInputLayerNeuronToOutputLayerNeuron()
        {
            SynapseWeightUpdateTester(learningRate: 1, momentum: 0)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0.23).Output(0.14).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0.92).Output(0.45).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.1)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 1,
                    synapseOutputNeuronId: 2,
                    expectedWeightDelta : 0.1288,
                    expectedWeight: 0.2288);
        }

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyHiddenLayerNeuronToHiddenLayerNeuron()
        {
             SynapseWeightUpdateTester(learningRate: 1, momentum: 0)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0.2123).Output(0.333).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0.1).Output(0.7451).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n =>  n.Id(3).ErrorRate(0.61234).Output(0.7812).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(4).ErrorRate(0.0012).Output(0.0122).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.23),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.89123),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.67123)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 2,
                    synapseOutputNeuronId: 3,
                    expectedWeightDelta : 0.456254534,
                    expectedWeight: 1.34748);
        }

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyHiddenLayerNeuronToOutputLayerNeuron()
        {
            SynapseWeightUpdateTester(learningRate: 1, momentum: 0)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0.2123).Output(0.333).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0.1).Output(0.7451).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n =>  n.Id(3).ErrorRate(0.61234).Output(0.7812).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(4).ErrorRate(0.0012).Output(0.0122).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.23),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.89123),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.67123)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 3,
                    synapseOutputNeuronId: 4,
                    expectedWeightDelta : 0.00093744,
                    expectedWeight: 0.67217);
        }
    }
}