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
                errorGradientDecimalPlaces: 5, 
                outputDecimalPlaces: 5, 
                synapseWeightDecimalPlaces: 5); 
        
        [Fact]
        public void CanFeedForwardCorrectlyInComplexMultiLayerNeuralNetwork()
        {
            FeedForwardTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(
                            n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(3).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(5).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(6).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(7).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(8).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(9).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(10).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(11).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(12).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(13).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(14).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(15).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(16).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(17).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(18).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(19).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(20).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 6, weight: 0.67),
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 7, weight: 0.72),
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 8, weight: 0.145),
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 9, weight: 0.134),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 6, weight: 0.999),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 7, weight: 0.13),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 8, weight: 0.2874),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 9, weight: 0.9),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 6, weight: 0.7823),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 7, weight: 0.234),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 8, weight: 0.981),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 9, weight: 0.99999),
                        s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 6, weight: 0.11),
                        s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 7, weight: 0.2687),
                        s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 8, weight: 0.6),
                        s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 9, weight: 0.614),
                        s => s.SynapseBetween(inputNeuronId: 5, outputNeuronId: 6, weight: 0.781),
                        s => s.SynapseBetween(inputNeuronId: 5, outputNeuronId: 7, weight: 0.7101),
                        s => s.SynapseBetween(inputNeuronId: 5, outputNeuronId: 8, weight: 0.9898),
                        s => s.SynapseBetween(inputNeuronId: 5, outputNeuronId: 9, weight: 0.3333),
                        s => s.SynapseBetween(inputNeuronId: 6, outputNeuronId: 10, weight: 0.1256),
                        s => s.SynapseBetween(inputNeuronId: 6, outputNeuronId: 11, weight: 0.876),
                        s => s.SynapseBetween(inputNeuronId: 6, outputNeuronId: 12, weight: 0.9999),
                        s => s.SynapseBetween(inputNeuronId: 7, outputNeuronId: 10, weight: 0.11333),
                        s => s.SynapseBetween(inputNeuronId: 7, outputNeuronId: 11, weight: 0.5555),
                        s => s.SynapseBetween(inputNeuronId: 7, outputNeuronId: 12, weight: 0.187),
                        s => s.SynapseBetween(inputNeuronId: 8, outputNeuronId: 10, weight: 0.7101),
                        s => s.SynapseBetween(inputNeuronId: 8, outputNeuronId: 11, weight: 0.68463),
                        s => s.SynapseBetween(inputNeuronId: 8, outputNeuronId: 12, weight: 0.123),
                        s => s.SynapseBetween(inputNeuronId: 9, outputNeuronId: 10, weight: 0.23412),
                        s => s.SynapseBetween(inputNeuronId: 9, outputNeuronId: 11, weight: 0.1),
                        s => s.SynapseBetween(inputNeuronId: 9, outputNeuronId: 12, weight: 0.2333),
                        s => s.SynapseBetween(inputNeuronId: 10, outputNeuronId: 13, weight: 0.9123),
                        s => s.SynapseBetween(inputNeuronId: 10, outputNeuronId: 14, weight: 0.9),
                        s => s.SynapseBetween(inputNeuronId: 10, outputNeuronId: 15, weight: 0.876),
                        s => s.SynapseBetween(inputNeuronId: 10, outputNeuronId: 16, weight: 0.7),
                        s => s.SynapseBetween(inputNeuronId: 11, outputNeuronId: 13, weight: 0.7101),
                        s => s.SynapseBetween(inputNeuronId: 11, outputNeuronId: 14, weight: 0.12457),
                        s => s.SynapseBetween(inputNeuronId: 11, outputNeuronId: 15, weight: 0.51),
                        s => s.SynapseBetween(inputNeuronId: 11, outputNeuronId: 16, weight: 0.4999),
                        s => s.SynapseBetween(inputNeuronId: 12, outputNeuronId: 13, weight: 0.3467),
                        s => s.SynapseBetween(inputNeuronId: 12, outputNeuronId: 14, weight: 0.9898),
                        s => s.SynapseBetween(inputNeuronId: 12, outputNeuronId: 15, weight: 0.1956),
                        s => s.SynapseBetween(inputNeuronId: 12, outputNeuronId: 16, weight: 0.9812),
                        s => s.SynapseBetween(inputNeuronId: 13, outputNeuronId: 17, weight: 0.2222),
                        s => s.SynapseBetween(inputNeuronId: 13, outputNeuronId: 18, weight: 0.2),
                        s => s.SynapseBetween(inputNeuronId: 13, outputNeuronId: 19, weight: 0.231),
                        s => s.SynapseBetween(inputNeuronId: 13, outputNeuronId: 20, weight: 0.7101),
                        s => s.SynapseBetween(inputNeuronId: 14, outputNeuronId: 17, weight: 0.4198),
                        s => s.SynapseBetween(inputNeuronId: 14, outputNeuronId: 18, weight: 0.7199),
                        s => s.SynapseBetween(inputNeuronId: 14, outputNeuronId: 19, weight: 0.1999),
                        s => s.SynapseBetween(inputNeuronId: 14, outputNeuronId: 20, weight: 0.5612),
                        s => s.SynapseBetween(inputNeuronId: 15, outputNeuronId: 17, weight: 0.78561),
                        s => s.SynapseBetween(inputNeuronId: 15, outputNeuronId: 18, weight: 0.1234),
                        s => s.SynapseBetween(inputNeuronId: 15, outputNeuronId: 19, weight: 0.6512),
                        s => s.SynapseBetween(inputNeuronId: 15, outputNeuronId: 20, weight: 0.9823),
                        s => s.SynapseBetween(inputNeuronId: 16, outputNeuronId: 17, weight: 0.9812),
                        s => s.SynapseBetween(inputNeuronId: 16, outputNeuronId: 18, weight: 0.1111),
                        s => s.SynapseBetween(inputNeuronId: 16, outputNeuronId: 19, weight: 0.3333),
                        s => s.SynapseBetween(inputNeuronId: 16, outputNeuronId: 20, weight: 0.7777)))
                .InputAndExpectOutput(
                    inputs: new [] { 0.9912, 0.81234, 0.0001, 0.45231, 0.67128 },
                    expectedOutput: new [] { 0.87728, 0.72048, 0.75779, 0.92166 });
        }

        [Fact]
        public void CanFeedForwardOutputCorrectlyForSingleInputNeuronNoHiddenLayersSingleOutputNeuron()
        {
             FeedForwardTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
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
                            n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(3).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
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
                            n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(3).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
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
                            n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(5).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
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
                            n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
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
                            n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(4).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
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
                            n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(3).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(4).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
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
                            n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
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
                            n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(3).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(4).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(5).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(6).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
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
            FeedForwardTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(
                            n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(5).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(6).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 3, weight: 0.67),
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 4, weight: 0.71),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.9812),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.45),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 5, weight: 0.99),
                        s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 5, weight: 0.127),
                        s => s.SynapseBetween(inputNeuronId: 5, outputNeuronId: 6, weight: 0.7101)))
                .InputAndExpectOutput(
                    inputs: new [] { 0.12345, 0.9812 },
                    expectedOutput: new [] { 0.62054 });
        }

        [Fact]
        public void CanFeedForwardCorrectlyMultipleInputNeuronsMultipleHiddenLayersMultipleOutputNeurons()
        {
            FeedForwardTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(
                            n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(5).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(6).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(7).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(8).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(9).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 3, weight: 0.67),
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 4, weight: 0.72),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.145),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.134),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 5, weight: 0.9),
                        s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 5, weight: 0.134),
                        s => s.SynapseBetween(inputNeuronId: 5, outputNeuronId: 6, weight: 0.11),
                        s => s.SynapseBetween(inputNeuronId: 6, outputNeuronId: 7, weight: 0.781),
                        s => s.SynapseBetween(inputNeuronId: 6, outputNeuronId: 8, weight: 0.7101),
                        s => s.SynapseBetween(inputNeuronId: 6, outputNeuronId: 9, weight: 0.7101)))
                .InputAndExpectOutput(
                    inputs: new [] { 0.9145, 0.7812 },
                    expectedOutput: new [] { 0.59985, 0.591, 0.591 });
        }
    }
}