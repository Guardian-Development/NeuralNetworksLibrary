using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Components.Activation.Functions;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class BackPropagationIntegrationTests : NeuralNetworkTest
    {
        private NeuralNetworkContext TestContext => 
            new NeuralNetworkContext(
                errorGradientDecimalPlaces: 5, 
                outputDecimalPlaces: 5, 
                synapseWeightDecimalPlaces: 5); 

        [Fact]
        public void CanTrainNoHiddenLayerNetworkForSingleEpoch()
        {
            BackPropagationTester.For(learningRate: 1.2, momentum: 0.2)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.15)))
                .QueueTrainingEpoch(e => e.Inputs(0.05).ExpectedOutputs(0.8712).ErrorRate(0.36933)
                    .ExpectNeuralNetworkState(nn => nn
                        .InputLayer(l => l
                            .Neurons(n => n.Id(1).ErrorGradient(0).Output(0.05)
                                .OutputSynapses(s => s.InputNeuronId(1).OutputNeuronId(2).Weight(0.15554))))
                        .OutputLayer(l => l
                            .Neurons(n => n.Id(2).ErrorGradient(0.09233).Output(0.50187)
                                .InputSynapses(s => s.InputNeuronId(1).OutputNeuronId(2).Weight(0.15554))))))
                .PerformAllEpochs();
        }

        [Fact(Skip="Not implemented")]
        public void CanTrainNoHiddenLayerNetworkForMultipleEpochs()
        {
        }

        [Fact(Skip="Not implemented")]
        public void CanTrainSingleHiddenLayerNetworkForSingleEpoch()
        {
        }

        [Fact(Skip="Not implemented")]
        public void CanTrainSingleHiddenLayerNetworkForMultipleEpochs()
        {
        }

        [Fact(Skip="Not implemented")]
        public void CanTrainMultipleHiddenLayerNetworkForSingleEpoch()
        {
        }

        [Fact(Skip="Not implemented")]
        public void CanTrainMultipleHiddenLayerNetworkForMultipleEpochs()
        {
        }
    }
}