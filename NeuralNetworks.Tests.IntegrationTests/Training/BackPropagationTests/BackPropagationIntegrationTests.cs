using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class BackPropagationIntegrationTests : NeuralNetworkTest
    {
        [Fact]
        public void CanTrainNoHiddenLayerNetworkForSingleEpoch()
        {
            BackPropagationTester.For(learningRate: 0.5, momentum: 1)
                .WithTargetNeuralNetwork(
                    nn => nn
                        .InputLayer(l => l
                            .Neuron(1, n => n.ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid))
                            .Neuron(2, n => n.ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid))
                            .BiasNeuron(3, n => n.ErrorRate(0).Output(1.00).Activation(ActivationType.Sigmoid)))
                        .OutputLayer(l => l
                            .Neuron(4, n => n.ErrorRate(0).Output(0.01).Activation(ActivationType.Sigmoid))
                            .Neuron(5, n => n.ErrorRate(0).Output(0.99).Activation(ActivationType.Sigmoid)))
                        .Synapses(ss => ss
                            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 4, weight: 0.15)
                            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 5, weight: 0.20)
                            .SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.25)
                            .SynapseBetween(inputNeuronId: 2, outputNeuronId: 5, weight: 0.30)
                            .SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 1.00)
                            .SynapseBetween(inputNeuronId: 3, outputNeuronId: 5, weight: 1.00))
                )
                .PerformTrainingEpoch(t => t.Inputs(0.05, 0.10).ExpectedOutputs(0.1, 0.9).ExpectedErrorRate(0.8)
                    .ExpectNeuralNetworkState(nn => nn
                        .ExpectedNeurons(
                            (1, n => n.ErrorRate(1).Output(1)),
                            (2, n => n.ErrorRate(1).Output(1)),
                            (3, n => n.ErrorRate(1).Output(1)),
                            (4, n => n.ErrorRate(1).Output(1)),
                            (5, n => n.ErrorRate(1).Output(1)))
                        .ExpectedSynapses(ss => ss
                            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 4, weight: 0.15)
                            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 5, weight: 0.20)
                            .SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.25)
                            .SynapseBetween(inputNeuronId: 2, outputNeuronId: 5, weight: 0.30)
                            .SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 1.00)
                            .SynapseBetween(inputNeuronId: 3, outputNeuronId: 5, weight: 1.00)))
                );
        }

        [Fact]
        public void CanTrainNoHiddenLayerNetworkForMultipleEpochs()
        {
        }

        [Fact]
        public void CanTrainSingleHiddenLayerNetworkForSingleEpoch()
        {
            BackPropagationTester.For(learningRate: 1, momentum: 2)
                .WithTargetNeuralNetwork(
                    nn => nn
                        .InputLayer(l => l
                            .Neuron(1, n => n.ErrorRate(1).Output(1).Activation(ActivationType.TanH))
                            .Neuron(2, n => n.ErrorRate(1).Output(1).Activation(ActivationType.TanH))
                            .BiasNeuron(3, n => n.ErrorRate(1).Output(1).Activation(ActivationType.TanH)))
                        .HiddenLayer(l => l
                            .Neuron(1, n => n.ErrorRate(1).Output(1).Activation(ActivationType.TanH))
                            .Neuron(2, n => n.ErrorRate(1).Output(1).Activation(ActivationType.TanH))
                            .BiasNeuron(3, n => n.ErrorRate(1).Output(1).Activation(ActivationType.TanH)))
                        .OutputLayer(l => l
                            .Neuron(1, n => n.ErrorRate(1).Output(1).Activation(ActivationType.TanH))
                            .Neuron(2, n => n.ErrorRate(1).Output(1).Activation(ActivationType.TanH)))
                        .Synapses(ss => ss
                            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 2)
                            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 2)
                            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 2)
                            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 2)
                            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 2))
                )
                .PerformTrainingEpoch(t => t.Inputs(1, 2).ExpectedOutputs(3).ExpectedErrorRate(0.8)
                    .ExpectNeuralNetworkState(nn => nn
                        .ExpectedNeurons(
                            (1, n => n.ErrorRate(1).Output(1)),
                            (2, n => n.ErrorRate(1).Output(1)),
                            (3, n => n.ErrorRate(1).Output(1)),
                            (4, n => n.ErrorRate(1).Output(1)),
                            (5, n => n.ErrorRate(1).Output(1)),
                            (6, n => n.ErrorRate(1).Output(1)))
                        .ExpectedSynapses(ss => ss
                            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 2)
                            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 2)
                            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 2)
                            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 2)
                            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 2)))
                );
        }

        [Fact]
        public void CanTrainSingleHiddenLayerNetworkForMultipleEpochs()
        {
        }

        [Fact]
        public void CanTrainMultipleHiddenLayerNetworkForSingleEpoch()
        {
        }

        [Fact]
        public void CanTrainMultipleHiddenLayerNetworkForMultipleEpochs()
        {
        }
    }
}