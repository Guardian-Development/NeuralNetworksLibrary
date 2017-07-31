
using System;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests.ErrorGradientCalculationsTests
{
    public sealed class BackPropagationErrorGradientCalculationTests : NeuralNetworkTest
    {
        public Func<BackPropagationErrorGradientTester> ErrorGradientTester 
            => BackPropagationErrorGradientTester.Create;
            
        private NeuralNetworkContext TestContext => 
            new NeuralNetworkContext(
                errorGradientDecimalPlaces: 5, 
                outputDecimalPlaces: 5, 
                synapseWeightDecimalPlaces: 5); 

        [Fact]
        public void CanComputeErrorGradientCorrectlyOutputLayerNeuron()
        {
             ErrorGradientTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0).Output(0.78).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorGradientForOutputLayerNeuron(
                    neuronId : 2, 
                    expectedOutput: 0.91,
                    errorGradient : 0.04835);

             ErrorGradientTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0).Output(0.12333).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorGradientForOutputLayerNeuron(
                    neuronId : 2, 
                    expectedOutput: 0.14,
                    errorGradient : -0.09733);

             ErrorGradientTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0).Output(0.55552).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorGradientForOutputLayerNeuron(
                    neuronId : 2, 
                    expectedOutput: 0.72,
                    errorGradient : 0.01959);
        }

        [Fact]
        public void CanComputeErrorGradientCorrectlyOutputLayerNeuronWhenErrorGradientIsZero()
        {
            ErrorGradientTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0).Output(0.55552).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorGradientForOutputLayerNeuron(
                    neuronId : 2, 
                    expectedOutput: 0.63542,
                    errorGradient : 0);
        }

        [Fact]
        public void CanComputeErrorGradientCorrectlyHiddenLayerNeuronDirectlyConnectedToOutputLayerWithSingleNeuron()
        {
             ErrorGradientTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0).Output(0.52224).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0).Output(0.62767).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(3).ErrorGradient(0.7821).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.11)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorGradientForHiddenLayerNeuron(
                    neuronId : 2,
                    errorGradient : 0.02011);

            ErrorGradientTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0).Output(0.52224).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0).Output(0.62767).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(3).ErrorGradient(-0.8912).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.01423)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorGradientForHiddenLayerNeuron(
                    neuronId : 2,
                    errorGradient : -0.00296);
        }

        [Fact]
        public void CanComputeErrorGradientCorrectlyHiddenLayerNeuronDirectlyConnectedToOutputLayerWithMultipleNeurons()
        {
            ErrorGradientTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0).Output(0.98123).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0).Output(0.72735).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorGradient(0.00123).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorGradient(0.55211).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(5).ErrorGradient(-0.0091).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.91),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.22154),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 5, weight: 0.0121)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorGradientForHiddenLayerNeuron(
                    neuronId : 2,
                    errorGradient : 0.02446);

             ErrorGradientTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0).Output(0.98123).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0).Output(0.72735).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorGradient(0.00123).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorGradient(-0.55211).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(5).ErrorGradient(0.0091).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.45),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.9812),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 5, weight: 0.03451)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorGradientForHiddenLayerNeuron(
                    neuronId : 2,
                    errorGradient : -0.10726);
        }

        [Fact]
        public void CanComputeErrorGradientCorrectlyHiddenLayerNeuronDirectlyConnectedToHiddenLayerWithSingleNeuron()
        {
            ErrorGradientTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0).Output(0.43218).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0).Output(0.60639).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(3).ErrorGradient(0.98121).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(4).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.1123),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorGradientForHiddenLayerNeuron(
                    neuronId : 2,
                    errorGradient : 0.0263);

             ErrorGradientTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0).Output(0.43218).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0).Output(0.60639).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(3).ErrorGradient(-0.45129).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(4).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.89891),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorGradientForHiddenLayerNeuron(
                    neuronId : 2,
                    errorGradient : -0.09683);
        }

        [Fact]
        public void CanComputeErrorGradientCorrectlyHiddenLayerNeuronDirectlyConnectedToHiddenLayerWithMultipleNeurons()
        {
             ErrorGradientTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0).Output(0.98451).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0).Output(0.728).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorGradient(0.22222).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorGradient(0.43129).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(5).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.47689),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.12121),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 5, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 5, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorGradientForHiddenLayerNeuron(
                    neuronId : 2,
                    errorGradient : 0.03134);

            ErrorGradientTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0).Output(0.98451).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0).Output(0.728).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorGradient(-0.12555).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorGradient(-0.99999).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(5).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.45121),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.99999),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 5, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 5, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorGradientForHiddenLayerNeuron(
                    neuronId : 2,
                    errorGradient : -0.20923);
        }

        [Fact]
        public void CanComputeErrorGradientCorrectlyHiddenLayerNeuronWhenErrorGradientIsZero()
        {
            ErrorGradientTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0).Output(0.98451).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0).Output(0.728).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorGradient(-0.12555).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(4).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(5).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.99999),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 5, weight: 1),
                        s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 5, weight: 1)))
                .ActivateNeuronWithId(neuronId: 2)
                .CalculateErrorGradientForHiddenLayerNeuron(
                    neuronId : 2,
                    errorGradient : 0);
        }
    }
}