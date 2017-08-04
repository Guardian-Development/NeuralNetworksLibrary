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
        private const int WeightDeltaAssertionPrecision = 9; 

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
                            .Neurons(
                                n => n.Id(1).ErrorGradient(0).Output(0.05)
                                    .OutputSynapses(s => 
                                        s.InputNeuronId(1).OutputNeuronId(2).Weight(0.15554)
                                                .WeightDelta(0.0055398, WeightDeltaAssertionPrecision))))
                        .OutputLayer(l => l
                            .Neurons(
                                n => n.Id(2).ErrorGradient(0.09233).Output(0.50187)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(2).Weight(0.15554)
                                                .WeightDelta(0.0055398, WeightDeltaAssertionPrecision))))))
                .PerformAllEpochs();
        }

        [Fact]
        public void CanTrainNoHiddenLayerNetworkForMultipleEpochs()
        {
             BackPropagationTester.For(learningRate: 1.72, momentum: 0.4)
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
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 3, weight: 0.34),
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 4, weight: 0.12),
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 5, weight: 0.98),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.64),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 0.14),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 5, weight: 0.39)))
                .QueueTrainingEpoch(e => e.Inputs(0.78, 0.98).ExpectedOutputs(0.78, 0.94, 0.12).ErrorRate(1.09205)
                    .ExpectNeuralNetworkState(nn => nn
                        .InputLayer(l => l
                            .Neurons(
                                n => n.Id(1).ErrorGradient(0).Output(0.78)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.35953)
                                            .WeightDelta(0.019533696, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(1).OutputNeuronId(4).Weight(0.24662)
                                            .WeightDelta(0.126620208, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(1).OutputNeuronId(5).Weight(0.82317)
                                            .WeightDelta(-0.15683304, WeightDeltaAssertionPrecision)), 
                                n => n.Id(2).ErrorGradient(0).Output(0.98)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.66454)
                                            .WeightDelta(0.024542336, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(4).Weight(0.29909)
                                            .WeightDelta(0.159086928, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(5).Weight(0.19295)
                                            .WeightDelta(-0.19704664, WeightDeltaAssertionPrecision))))
                        .OutputLayer(l => l
                            .Neurons(
                                n => n.Id(3).ErrorGradient(0.01456).Output(0.70939)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.35953)
                                            .WeightDelta(0.019533696, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.66454)
                                            .WeightDelta(0.024542336, WeightDeltaAssertionPrecision)),
                                n => n.Id(4).ErrorGradient(0.09438).Output(0.55745)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(4).Weight(0.24662)
                                            .WeightDelta(0.126620208, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(4).Weight(0.29909)
                                            .WeightDelta(0.159086928, WeightDeltaAssertionPrecision)),
                                n => n.Id(5).ErrorGradient(-0.1169).Output(0.75889)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(5).Weight(0.82317)
                                            .WeightDelta(-0.15683304, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(5).Weight(0.19295)
                                            .WeightDelta(-0.19704664, WeightDeltaAssertionPrecision))))))
                .PerformAllEpochs();
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