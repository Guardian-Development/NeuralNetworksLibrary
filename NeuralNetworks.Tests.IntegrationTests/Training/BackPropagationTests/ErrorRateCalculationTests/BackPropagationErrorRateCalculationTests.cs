
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

        [Fact]
        public void CanComputeErrorRateCorrectlyHiddenLayerNeuronDirectlyConnectedToOutputLayerWithSingleNeuron()
        {
             ErrorRateTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0.52224).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0.62767).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(3).ErrorRate(0.7821).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.11)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorRateForHiddenLayerNeuron(
                    neuronId : 2,
                    errorRate : 0.02011);

            ErrorRateTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0.52224).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0.62767).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(3).ErrorRate(-0.8912).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.01423)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorRateForHiddenLayerNeuron(
                    neuronId : 2,
                    errorRate : -0.00296);
        }

        [Fact]
        public void CanComputeErrorRateCorrectlyHiddenLayerNeuronDirectlyConnectedToOutputLayerWithMultipleNeurons()
        {
            ErrorRateTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0.98123).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0.72735).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorRate(0.00123).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorRate(0.55211).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(5).ErrorRate(-0.0091).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.91),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.22154),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 5, weight: 0.0121)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorRateForHiddenLayerNeuron(
                    neuronId : 2,
                    errorRate : 0.02446);

             ErrorRateTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0.98123).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0.72735).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorRate(0.00123).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorRate(-0.55211).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(5).ErrorRate(0.0091).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.45),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.9812),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 5, weight: 0.03451)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorRateForHiddenLayerNeuron(
                    neuronId : 2,
                    errorRate : -0.10726);
        }

        [Fact]
        public void CanComputeErrorRateCorrectlyHiddenLayerNeuronDirectlyConnectedToHiddenLayerWithSingleNeuron()
        {
            ErrorRateTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0.43218).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0.60639).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(3).ErrorRate(0.98121).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(4).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.1123),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorRateForHiddenLayerNeuron(
                    neuronId : 2,
                    errorRate : 0.0263);

             ErrorRateTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0.43218).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0.60639).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(3).ErrorRate(-0.45129).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(4).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.89891),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorRateForHiddenLayerNeuron(
                    neuronId : 2,
                    errorRate : -0.09683);
        }

        [Fact]
        public void CanComputeErrorRateCorrectlyHiddenLayerNeuronDirectlyConnectedToHiddenLayerWithMultipleNeurons()
        {
             ErrorRateTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0.98451).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0.728).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorRate(0.22222).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorRate(0.43129).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(5).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.47689),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.12121),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 5, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 5, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorRateForHiddenLayerNeuron(
                    neuronId : 2,
                    errorRate : 0.03134);

            ErrorRateTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0.98451).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0.728).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorRate(-0.12555).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorRate(-0.99999).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(5).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.45121),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.99999),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 5, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 5, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorRateForHiddenLayerNeuron(
                    neuronId : 2,
                    errorRate : -0.20923);
        }

        [Fact]
        public void CanComputeErrorRateCorrectlyHiddenLayerNeuronWhenErrorRateIsZero()
        {
            ErrorRateTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0.98451).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0.728).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorRate(-0.12555).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(5).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.99999),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 5, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 5, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorRateForHiddenLayerNeuron(
                    neuronId : 2,
                    errorRate : 0);
        }
    }
}