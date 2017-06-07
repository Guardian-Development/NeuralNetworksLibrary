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