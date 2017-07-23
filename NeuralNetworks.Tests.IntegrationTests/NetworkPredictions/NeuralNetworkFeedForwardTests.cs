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
        public void CanFeedForwardOutputCorrectlyForSingleInputNeuronNoHiddenLayersSingleOutputNeuron()
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
                    expectedOutput: new [] { 0.50375 });
        }

        [Fact]
        public void CanFeedForwardOutputCorrectlyForMultipleInputNeuronsNoHiddenLayersSingleOutputNeuron()
        {
             FeedForwardTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(
                            n => n.Id(1).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(3).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 3, weight: 0.87),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.11)))
                .InputAndExpectOutput(
                    inputs: new [] { 0.4, 0.56 },
                    expectedOutput: new [] { 0.60099 });
        }

        [Fact]
        public void CanFeedForwardOutputCorrectlyForSingleInputNeuronNoHiddenLayersMultipleOutputNeurons()
        {
            FeedForwardTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(
                            n => n.Id(1).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(3).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.45),
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 3, weight: 0.32)))
                .InputAndExpectOutput(
                    inputs: new [] { 0.39845 },
                    expectedOutput: new [] { 0.54471, 0.53183 });
        }

        [Fact]
        public void CanFeedForwardCorrectlyForMultipleInputNeuronsNoHiddenLayersMultipleOutputNeurons()
        {
             FeedForwardTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(
                            n => n.Id(1).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(5).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 3, weight: 0.01),
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 4, weight: 0.99),
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 5, weight: 0.56),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.76),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.21),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 5, weight: 0.33)))
                .InputAndExpectOutput(
                    inputs: new [] { 0.987, 0.32178 },
                    expectedOutput: new [] { 0.56326, 0.73976, 0.65901 });
        }

        [Fact]
        public void CanFeedForwardCorrectlySingleInputNeuronSingleHiddenLayerWithSingleNeuronSingleOutputNeuron()
        {
            FeedForwardTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(
                            n => n.Id(1).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.00001),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.76543)))
                .InputAndExpectOutput(
                    inputs: new [] { 0.439 },
                    expectedOutput: new [] { 0.59453 });
        }

        [Fact]
        public void CanFeedForwardCorrectlyMultipleInputNeuronSingleHiddenLayerWithSingleNeuronSingleOutputNeuron()
        {
            FeedForwardTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(
                            n => n.Id(1).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(4).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 3, weight: 0.76555),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.99999),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.89333)))
                .InputAndExpectOutput(
                    inputs: new [] { 0.333, 0.12567 },
                    expectedOutput: new [] { 0.62964 });
        }

        [Fact]
        public void CanFeedForwardCorrectlySingleInputNeuronSingleHiddenLayerWithMultipleNeuronsSingleOutputNeuron()
        {
            FeedForwardTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(
                            n => n.Id(1).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(3).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(4).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.87129),
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 3, weight: 0.3985),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.1267),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.6523)))
                .InputAndExpectOutput(
                    inputs: new [] { 0.9912348 },
                    expectedOutput: new [] { 0.61748 });
        }

        [Fact]
        public void CanFeedForwardCorrectlySingleInputNeuronSingleHiddenLayerWithSingleNeuronMultipleOutputNeurons()
        {
            FeedForwardTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(
                            n => n.Id(1).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.9812),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.6209)))
                .InputAndExpectOutput(
                    inputs: new [] { 0.000102 },
                    expectedOutput: new [] { 0.5125, 0.577 });
        }

        [Fact]
        public void CanFeedForwardCorrectlySingleInputNeuronMultipleHiddenLayersSingleOutputNeuron()
        {
            FeedForwardTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(
                            n => n.Id(1).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(3).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(4).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(5).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(6).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.1327),
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 3, weight: 0.823),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.691),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.5),
                        s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 5, weight: 0.8723),
                        s => s.SynapseBetween(inputNeuronId: 5, outputNeuronId: 6, weight: 0.0034)))
                .InputAndExpectOutput(
                    inputs: new [] { 0.98124 },
                    expectedOutput: new [] { 0.50055 });
        }

        [Fact]
        public void CanFeedForwardCorrectlyMultipleInputNeuronsMultipleHiddenLayersSingleOutputNeuron()
        {

        }

        [Fact]
        public void CanFeedForwardCorrectlySingleInputNeuronsMultipleHiddenLayersSingleOutputNeuron()
        {

        }

        [Fact]
        public void CanFeedForwardCorrectlyMultipleInputNeuronsMultipleHiddenLayersMultipleOutputNeurons()
        {

        }

        [Fact]
        public void CanFeedForwardCorrectlyInComplexMultiLayeredNeuralNetwork()
        {

        }
    }
}