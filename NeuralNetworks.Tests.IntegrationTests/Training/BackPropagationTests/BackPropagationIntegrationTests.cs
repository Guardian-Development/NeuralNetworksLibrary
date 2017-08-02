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
                                                .WeightDelta(0.0053398, WeightDeltaAssertionPrecision))))
                        .OutputLayer(l => l
                            .Neurons(
                                n => n.Id(2).ErrorGradient(0.09233).Output(0.50187)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(2).Weight(0.15554)
                                                .WeightDelta(0.0053398, WeightDeltaAssertionPrecision))))))
                .PerformAllEpochs();
        }

        [Fact(Skip="Numbers need rework")]
        public void CanTrainNoHiddenLayerNetworkForMultipleEpochs()
        {
            //calculating error rates is wrong in this test - investigate.
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
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.3491)
                                            .WeightDelta(0.009096048, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(1).OutputNeuronId(4).Weight(0.21111)
                                            .WeightDelta(0.091108056, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(1).OutputNeuronId(5).Weight(1.12407)
                                            .WeightDelta(0.144074424, WeightDeltaAssertionPrecision)), 
                                n => n.Id(2).ErrorGradient(0).Output(0.98)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.65143)
                                            .WeightDelta(0.011428368, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(4).Weight(0.25447)
                                            .WeightDelta(0.114469096, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(5).Weight(0.57102)
                                            .WeightDelta(0.181016584, WeightDeltaAssertionPrecision))))
                        .OutputLayer(l => l
                            .Neurons(
                                n => n.Id(3).ErrorGradient(0.00678).Output(0.70939)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.3491)
                                            .WeightDelta(0.009096048, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.65143)
                                            .WeightDelta(0.011428368, WeightDeltaAssertionPrecision)),
                                n => n.Id(4).ErrorGradient(0.06791).Output(0.55745)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(4).Weight(0.21111)
                                            .WeightDelta(0.091108056, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(4).Weight(0.25447)
                                            .WeightDelta(0.114469096, WeightDeltaAssertionPrecision)),
                                n => n.Id(5).ErrorGradient(0.10739).Output(0.75889)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(5).Weight(1.12407)
                                            .WeightDelta(0.144074424, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(5).Weight(0.57102)
                                            .WeightDelta(0.181016584, WeightDeltaAssertionPrecision))))))
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