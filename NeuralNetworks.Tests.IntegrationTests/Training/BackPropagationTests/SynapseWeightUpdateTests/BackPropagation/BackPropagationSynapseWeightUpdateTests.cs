using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests.SynapseWeightUpdateTests.BackPropagation
{
    public sealed class BackPropagationSynapseWeightUpdateTests : NeuralNetworkTest
    {
        public BackPropagationSynapseWeightUpdateTester SynapseWeightUpdateTester(double learningRate, double momentum)
            => BackPropagationSynapseWeightUpdateTester.ForBackPropagation(learningRate, momentum);
            
        private NeuralNetworkContext TestContext => 
            new NeuralNetworkContext(
                errorGradientDecimalPlaces: 5, 
                outputDecimalPlaces: 5, 
                synapseWeightDecimalPlaces: 5); 

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyInputLayerNeuronToHiddenLayerNeuron()
        {
             SynapseWeightUpdateTester(learningRate: 0.781, momentum: 0.89)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.15).Output(0.78).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(-0.16).Output(0.91).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(3).ErrorGradient(0.98).Output(0.77).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.1423).WithWeightDelta(0.131),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.11)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 1,
                    synapseOutputNeuronId: 2,
                    expectedWeightDelta : -0.0974688,
                    expectedWeight: 0.16142);
        }

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyInputLayerNeuronToOutputLayerNeuron()
        {
            SynapseWeightUpdateTester(learningRate: 1.981, momentum: 1.0023)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.156).Output(0.372).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.0012).Output(0.071).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.0023).WithWeightDelta(0.0213)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 1,
                    synapseOutputNeuronId: 2,
                    expectedWeightDelta : 0.000884318,
                    expectedWeight: 0.02453);
        }

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyHiddenLayerNeuronToHiddenLayerNeuron()
        {
             SynapseWeightUpdateTester(learningRate: 0.981, momentum: 1.0452)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.2123).Output(0.333).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.342).Output(0.898).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n =>  n.Id(3).ErrorGradient(-1.234).Output(0.87).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(4).ErrorGradient(0.0012).Output(0.0122).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.23),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.82111).WithWeightDelta(-0.0023),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.67123)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 2,
                    synapseOutputNeuronId: 3,
                    expectedWeightDelta : -1.087077492,
                    expectedWeight: -0.26837);
        }

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyHiddenLayerNeuronToOutputLayerNeuron()
        {
            SynapseWeightUpdateTester(learningRate: 1.212, momentum: 1.232)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.2123).Output(0.333).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.1).Output(0.7451).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n =>  n.Id(3).ErrorGradient(0.61234).Output(0.6523).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(4).ErrorGradient(0.341).Output(0.0122).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.23),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.89123),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.762).WithWeightDelta(0.3412)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 3,
                    synapseOutputNeuronId: 4,
                    expectedWeightDelta : 0.269590372,
                    expectedWeight: 1.45195);
        } 
    }
}