using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class BackPropagationIntegrationTests : NeuralNetworkTest
    {
        [Fact]
        public void CanTrainNoHiddenLayerSingleInputNeuronSingleOutputNeuronNetworkForSingleEpoch()
        {
                //BackPropagationTester.For(learningRate: 0.5, momentum: 0)
                //.WithTargetNeuralNetwork(
                //    nn => nn
                //        .Context(errorRateDecimalPlaces: 10, outputDecimalPlaces: 10, synapseWeightDecimalPlaces: 10)
                //        .InputLayer(l => l
                //            .Neuron(1, n => n.ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                //        .OutputLayer(l => l
                //            .Neuron(2, n => n.ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                //        .Synapses(ss => ss
                //            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.15))
                //)
                //.PerformTrainingEpoch(t => t.Inputs(0.05).ExpectedOutputs(0.1).ExpectedErrorRate(0.8)
                //    .ExpectNeuralNetworkState(nn => nn
                //        .ExpectedNeurons(
                //            (1, n => n.ErrorRate(0).Output(0.05)),
                //            (2, n => n.ErrorRate(0.08075175428).Output(0.50187499121098693819516206063481)))
                //        .ExpectedSynapses(ss => ss
                //            .SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.1496232475)))
                //);

            BackPropagationTester.For(learningRate: 0.5, momentum: 0)
			    .WithTargetNeuralNetwork(
                     NeuralNetworkContext.MaximumPrecision,
                     nn => nn
						.InputLayer(l => l
							.Neuron(1, n => n.ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
						.OutputLayer(l => l
							.Neuron(2, n => n.ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
						.Synapses(ss => ss
							.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.15))
				)
				.PerformTrainingEpoch(t => t.Inputs(0.05).ExpectedOutputs(0.1).ExpectedErrorRate(0.8)
				  .ExpectNeuralNetwork(nn => nn
						.ExpectInputLayer(l => l
							.ExpectNeuron(n => n.Id(1).Error(1).Output(1).InputSynapses(s => s.Weight(1), s => s.Weight(1))
							.ExpectNeuron(n => n.Id(1).Error(1).Output(1).InputSynapses(s => s.Weight(1), s => s.Weight(1)))))
						.ExpectHiddenLayer(l => l
							.ExpectNeuron(n => n.Id(1).Error(1).Output(1).InputSynapses(s => s.Weight(1), s => s.Weight(1))
							.ExpectNeuron(n => n.Id(1).Error(1).Output(1).InputSynapses(s => s.Weight(1), s => s.Weight(1)))))
						.ExpectHiddenLayer(l => l
							.ExpectNeuron(n => n.Id(1).Error(1).Output(1).InputSynapses(s => s.Weight(1), s => s.Weight(1))
							.ExpectNeuron(n => n.Id(1).Error(1).Output(1).InputSynapses(s => s.Weight(1), s => s.Weight(1)))))
						.ExpectOutputLayer(l => l
							.ExpectNeuron(n => n.Id(1).Error(1).Output(1).InputSynapses(s => s.Weight(1), s => s.Weight(1))
							.ExpectNeuron(n => n.Id(1).Error(1).Output(1).InputSynapses(s => s.Weight(1), s => s.Weight(1)))))));
        }

        [Fact]
        public void CanTrainNoHiddenLayerNetworkForMultipleEpochs()
        {
        }

        [Fact]
        public void CanTrainSingleHiddenLayerNetworkForSingleEpoch()
        {
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