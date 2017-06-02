using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class BackPropagationIntegrationTests : NeuralNetworkTest
    {
        [Fact]
        public void CanTrainNetworkForSingleEpoch()
        {
            BackPropagationTester.For(learningRate: 1, momentum: 2)
                .WithInitialNeuralNetwork(
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
                            .SynapseBetween(s => s.InputNeuronId(1).OutputNeuronId(2).Weight(2))
                            .SynapseBetween(s => s.InputNeuronId(1).OutputNeuronId(3).Weight(2))
                            .SynapseBetween(s => s.InputNeuronId(2).OutputNeuronId(4).Weight(2))
                            .SynapseBetween(s => s.InputNeuronId(2).OutputNeuronId(4).Weight(2))
                            .SynapseBetween(s => s.InputNeuronId(3).OutputNeuronId(7).Weight(2)))
                )
                .PerformTrainingEpoch(t => t.Inputs(1, 2).ExpectedOutputs(3).ExpectedErrorRate(0.8)
                    .ExpectNeuralNetworkState(nn => nn
                        .ExpectedNeurons(
                            n => n.Id(1).ErrorRate(1).Output(1),
                            n => n.Id(2).ErrorRate(1).Output(1),
                            n => n.Id(3).ErrorRate(1).Output(1),
                            n => n.Id(4).ErrorRate(1).Output(1),
                            n => n.Id(5).ErrorRate(1).Output(1),
                            n => n.Id(6).ErrorRate(1).Output(1))
                        .ExpectedSynapses(
                            s => s.InputNeuronId(1).OutputNeuronId(2).Weight(2),
                            s => s.InputNeuronId(1).OutputNeuronId(2).Weight(2),
                            s => s.InputNeuronId(1).OutputNeuronId(2).Weight(2),
                            s => s.InputNeuronId(1).OutputNeuronId(2).Weight(2),
                            s => s.InputNeuronId(1).OutputNeuronId(2).Weight(2),
                            s => s.InputNeuronId(1).OutputNeuronId(2).Weight(2),
                            s => s.InputNeuronId(1).OutputNeuronId(2).Weight(2)))
                );
        }
    }
}