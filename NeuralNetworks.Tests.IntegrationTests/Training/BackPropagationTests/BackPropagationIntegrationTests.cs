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
                errorRateDecimalPlaces: 5, 
                outputDecimalPlaces: 5, 
                synapseWeightDecimalPlaces: 5); 

        private IProvideRandomNumberGeneration PredictableGenerator => 
            PredictableRandomNumberGenerator.Create(); 

        [Fact]
        public void CanTrainNoHiddenLayerSingleInputNeuronSingleOutputNeuronNetworkForSingleEpoch()
        {
            BackPropagationTester.For(learningRate: 0.5, momentum: 0)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorRate(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.15)))
                .QueueTrainingEpoch(e => e.Inputs(0.05).ExpectedOutputs(1).ErrorRate(0.12407)
                    .ExpectNeuralNetworkState(nn => nn
                        .InputLayer(l => l
                            .Neurons(n => 
                                n.Id(1).ErrorRate(0).Output(0.05)
                                .OutputSynapses(s => s.InputNeuronId(1).OutputNeuronId(2).Weight(0.15311))))
                        .OutputLayer(l => l
                            .Neurons(n => 
                                n.Id(2).ErrorRate(0.12407).Output(0.50187)
                                .InputSynapses(s => s.InputNeuronId(1).OutputNeuronId(2).Weight(0.15311))))))
                .PerformAllEpochs();
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