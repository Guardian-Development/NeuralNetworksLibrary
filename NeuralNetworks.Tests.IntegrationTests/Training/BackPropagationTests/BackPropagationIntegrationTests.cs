using System.Threading.Tasks;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Tests.Support;
using NeuralNetworks.Library.Extensions; 
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
                                                .WeightDelta(0.0055398, WeightDeltaAssertionPrecision))))), 
                    ParallelOptionsExtensions.MultiThreadedOptions(10))
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
                                            .WeightDelta(-0.19704664, WeightDeltaAssertionPrecision))))),
                    ParallelOptionsExtensions.SingleThreadedOptions())
                .QueueTrainingEpoch(e => e.Inputs(0.78, 0.98).ExpectedOutputs(0.78, 0.94, 0.12).ErrorRate(0.96014)
                    .ExpectNeuralNetworkState(nn => nn
                        .InputLayer(l => l
                            .Neurons(
                                n => n.Id(1).ErrorGradient(0).Output(0.78)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.38437)
                                            .WeightDelta(0.017024904, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(1).OutputNeuronId(4).Weight(0.39881)
                                            .WeightDelta(0.101545704, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(1).OutputNeuronId(5).Weight(0.59695)
                                            .WeightDelta(-0.163487376, WeightDeltaAssertionPrecision)), 
                                n => n.Id(2).ErrorGradient(0).Output(0.98)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.69575)
                                            .WeightDelta(0.021390264, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(4).Weight(0.49031)
                                            .WeightDelta(0.127583064, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(5).Weight(-0.09128)
                                            .WeightDelta(-0.205407216, WeightDeltaAssertionPrecision))))
                        .OutputLayer(l => l
                            .Neurons(
                                n => n.Id(3).ErrorGradient(0.01269).Output(0.71742)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.38437)
                                            .WeightDelta(0.017024904, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.69575)
                                            .WeightDelta(0.021390264, WeightDeltaAssertionPrecision)),
                                n => n.Id(4).ErrorGradient(0.07569).Output(0.61904)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(4).Weight(0.39881)
                                            .WeightDelta(0.101545704, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(4).Weight(0.49031)
                                            .WeightDelta(0.127583064, WeightDeltaAssertionPrecision)),
                                n => n.Id(5).ErrorGradient(-0.12186).Output(0.6966)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(5).Weight(0.59695)
                                            .WeightDelta(-0.163487376, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(5).Weight(-0.09128)
                                            .WeightDelta(-0.205407216, WeightDeltaAssertionPrecision))))),
                    ParallelOptionsExtensions.MultiThreadedOptions(10))
                .PerformAllEpochs();
        }

        [Fact]
        public void CanTrainSingleHiddenLayerNetworkForSingleEpoch()
        {
            BackPropagationTester.For(learningRate: 2.1, momentum: 0.32)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(
                            n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(4).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(5).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 3, weight: 0.231),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.4513),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.215),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 5, weight: 0.61)))
                .QueueTrainingEpoch(e => e.Inputs(0.1432, 0.4513).ExpectedOutputs(0.23, 0.167).ErrorRate(0.71743)
                    .ExpectNeuralNetworkState(nn => nn
                        .InputLayer(l => l
                            .Neurons(
                                n => n.Id(1).ErrorGradient(0).Output(0.1432)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.22522)
                                            .WeightDelta(-0.005776831, WeightDeltaAssertionPrecision)), 
                                n => n.Id(2).ErrorGradient(0).Output(0.4513)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.43309)
                                            .WeightDelta(-0.018205893, WeightDeltaAssertionPrecision))))
                        .HiddenLayers(l => l
                            .Neurons(
                                n => n.Id(3).ErrorGradient(-0.01921).Output(0.55891)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.22522)
                                            .WeightDelta(-0.005776831, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.43309)
                                            .WeightDelta(-0.018205893, WeightDeltaAssertionPrecision))
                                    .OutputSynapses(
                                        s => s.InputNeuronId(3).OutputNeuronId(4).Weight(0.12729)
                                            .WeightDelta(-0.087711423, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(3).OutputNeuronId(5).Weight(0.49101)
                                            .WeightDelta(-0.118990821, WeightDeltaAssertionPrecision)))) 
                        .OutputLayer(l => l
                            .Neurons(
                                n => n.Id(4).ErrorGradient(-0.07473).Output(0.53001)
                                    .InputSynapses(
                                        s => s.InputNeuronId(3).OutputNeuronId(4).Weight(0.12729)
                                            .WeightDelta(-0.087711423, WeightDeltaAssertionPrecision)),
                                n => n.Id(5).ErrorGradient(-0.10138).Output(0.58442)
                                    .InputSynapses(
                                        s => s.InputNeuronId(3).OutputNeuronId(5).Weight(0.49101)
                                            .WeightDelta(-0.118990821, WeightDeltaAssertionPrecision))))),
                    ParallelOptionsExtensions.MultiThreadedOptions(10))
                .PerformAllEpochs();
        }

        [Fact]
        public void CanTrainSingleHiddenLayerNetworkForMultipleEpochs()
        {
            BackPropagationTester.For(learningRate: 1.021, momentum: 0.416)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(
                            n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(4).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(5).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 3, weight: 0.156),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.212),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.89123),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 5, weight: 0.98)))
                .QueueTrainingEpoch(e => e.Inputs(0.1432, 0.4513).ExpectedOutputs(0.23, 0.167).ErrorRate(0.84572)
                    .ExpectNeuralNetworkState(nn => nn
                        .InputLayer(l => l
                            .Neurons(
                                n => n.Id(1).ErrorGradient(0).Output(0.1432)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.1492)
                                            .WeightDelta(-0.006803021, WeightDeltaAssertionPrecision)), 
                                n => n.Id(2).ErrorGradient(0).Output(0.4513)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.19056)
                                            .WeightDelta(-0.021439968, WeightDeltaAssertionPrecision))))
                        .HiddenLayers(l => l
                            .Neurons(
                                n => n.Id(3).ErrorGradient(-0.04653).Output(0.52947)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.1492)
                                            .WeightDelta(-0.006803021, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.19056)
                                            .WeightDelta(-0.021439968, WeightDeltaAssertionPrecision))
                                    .OutputSynapses(
                                        s => s.InputNeuronId(3).OutputNeuronId(4).Weight(0.84189)
                                            .WeightDelta(-0.049344952, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(3).OutputNeuronId(5).Weight(0.92185)
                                            .WeightDelta(-0.058151145, WeightDeltaAssertionPrecision)))) 
                        .OutputLayer(l => l
                            .Neurons(
                                n => n.Id(4).ErrorGradient(-0.09128).Output(0.61583)
                                    .InputSynapses(
                                        s => s.InputNeuronId(3).OutputNeuronId(4).Weight(0.84189)
                                            .WeightDelta(-0.049344952, WeightDeltaAssertionPrecision)),
                                n => n.Id(5).ErrorGradient(-0.10757).Output(0.62689)
                                    .InputSynapses(
                                        s => s.InputNeuronId(3).OutputNeuronId(5).Weight(0.92185)
                                            .WeightDelta(-0.058151145, WeightDeltaAssertionPrecision))))),
                    ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions())
                .QueueTrainingEpoch(e => e.Inputs(0.1432, 0.4513).ExpectedOutputs(0.23, 0.167).ErrorRate(0.83118)
                    .ExpectNeuralNetworkState(nn => nn
                        .InputLayer(l => l
                            .Neurons(
                                n => n.Id(1).ErrorGradient(0).Output(0.1432)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.14002)
                                            .WeightDelta(-0.006351241, WeightDeltaAssertionPrecision)), 
                                n => n.Id(2).ErrorGradient(0).Output(0.4513)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.16162)
                                            .WeightDelta(-0.020016166, WeightDeltaAssertionPrecision))))
                        .HiddenLayers(l => l
                            .Neurons(
                                n => n.Id(3).ErrorGradient(-0.04344).Output(0.52682)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.14002)
                                            .WeightDelta(-0.006351241, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.16162)
                                            .WeightDelta(-0.020016166, WeightDeltaAssertionPrecision))
                                    .OutputSynapses(
                                        s => s.InputNeuronId(3).OutputNeuronId(4).Weight(0.77281)
                                            .WeightDelta(-0.048549339, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(3).OutputNeuronId(5).Weight(0.84032)
                                            .WeightDelta(-0.05734373, WeightDeltaAssertionPrecision)))) 
                        .OutputLayer(l => l
                            .Neurons(
                                n => n.Id(4).ErrorGradient(-0.09026).Output(0.6091)
                                    .InputSynapses(
                                        s => s.InputNeuronId(3).OutputNeuronId(4).Weight(0.77281)
                                            .WeightDelta(-0.048549339, WeightDeltaAssertionPrecision)),
                                n => n.Id(5).ErrorGradient(-0.10661).Output(0.61908)
                                    .InputSynapses(
                                        s => s.InputNeuronId(3).OutputNeuronId(5).Weight(0.84032)
                                            .WeightDelta(-0.05734373, WeightDeltaAssertionPrecision))))),
                    ParallelOptionsExtensions.MultiThreadedOptions(10))
                .PerformAllEpochs();
        }

        [Fact]
        public void CanTrainMultipleHiddenLayerNetworkForSingleEpoch()
        {
            BackPropagationTester.For(learningRate: 1.834, momentum: 0.814)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(
                            n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(4).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(5).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(6).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 3, weight: 0.712),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.682),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.991),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 5, weight: 0.78123),
                        s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 6, weight: 0.5564),
                        s => s.SynapseBetween(inputNeuronId: 5, outputNeuronId: 6, weight: 0.121)))
                .QueueTrainingEpoch(e => e.Inputs(0.7123, 0.912).ExpectedOutputs(0.991).ErrorRate(0.37903)
                    .ExpectNeuralNetworkState(nn => nn
                        .InputLayer(l => l
                            .Neurons(
                                n => n.Id(1).ErrorGradient(0).Output(0.7123)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.71508)
                                            .WeightDelta(0.003083005, WeightDeltaAssertionPrecision)), 
                                n => n.Id(2).ErrorGradient(0).Output(0.912)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.68595)
                                            .WeightDelta(0.003947355, WeightDeltaAssertionPrecision))))
                        .HiddenLayers(
                            l => l.Neurons(
                                    n => n.Id(3).ErrorGradient(0.00236).Output(0.75568)
                                        .InputSynapses(
                                            s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.71508)
                                                .WeightDelta(0.003083005, WeightDeltaAssertionPrecision),
                                            s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.68595)
                                                .WeightDelta(0.003947355, WeightDeltaAssertionPrecision))
                                        .OutputSynapses(
                                            s => s.InputNeuronId(3).OutputNeuronId(4).Weight(1.00613)
                                                .WeightDelta(0.015134215, WeightDeltaAssertionPrecision),
                                            s => s.InputNeuronId(3).OutputNeuronId(5).Weight(0.78469)
                                                .WeightDelta(0.003464793, WeightDeltaAssertionPrecision))),
                            l => l.Neurons(
                                    n => n.Id(4).ErrorGradient(0.01092).Output(0.67893)
                                        .InputSynapses(
                                            s => s.InputNeuronId(3).OutputNeuronId(4).Weight(1.00613)
                                                .WeightDelta(0.015134215, WeightDeltaAssertionPrecision))
                                        .OutputSynapses(
                                            s => s.InputNeuronId(4).OutputNeuronId(6).Weight(0.66848)
                                                .WeightDelta(0.112076637, WeightDeltaAssertionPrecision)),
                                     n => n.Id(5).ErrorGradient(0.0025).Output(0.64345)
                                        .InputSynapses(
                                            s => s.InputNeuronId(3).OutputNeuronId(5).Weight(0.78469)
                                                .WeightDelta(0.003464793, WeightDeltaAssertionPrecision))
                                        .OutputSynapses(
                                            s => s.InputNeuronId(5).OutputNeuronId(6).Weight(0.22722)
                                                .WeightDelta(0.106219658, WeightDeltaAssertionPrecision)))) 
                        .OutputLayer(l => l
                            .Neurons(
                                n => n.Id(6).ErrorGradient(0.09001).Output(0.61197)
                                    .InputSynapses(
                                        s => s.InputNeuronId(4).OutputNeuronId(6).Weight(0.66848)
                                            .WeightDelta(0.112076637, WeightDeltaAssertionPrecision),
                                         s => s.InputNeuronId(5).OutputNeuronId(6).Weight(0.22722)
                                            .WeightDelta(0.106219658, WeightDeltaAssertionPrecision))))),                      
                    ParallelOptionsExtensions.SingleThreadedOptions())
                .PerformAllEpochs();
        }

        [Fact]
        public void CanTrainMultipleHiddenLayerNetworkForMultipleEpochs()
        {
            BackPropagationTester.For(learningRate: 1.834, momentum: 0.814)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(
                            n => n.Id(1).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(2).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(3).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(
                            n => n.Id(4).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid),
                            n => n.Id(5).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(
                            n => n.Id(6).ErrorGradient(0).Output(0).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 3, weight: 0.712),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.682),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.991),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 5, weight: 0.78123),
                        s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 6, weight: 0.5564),
                        s => s.SynapseBetween(inputNeuronId: 5, outputNeuronId: 6, weight: 0.121)))
                .QueueTrainingEpoch(e => e.Inputs(0.7123, 0.912).ExpectedOutputs(0.991).ErrorRate(0.37903)
                    .ExpectNeuralNetworkState(nn => nn
                        .InputLayer(l => l
                            .Neurons(
                                n => n.Id(1).ErrorGradient(0).Output(0.7123)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.71508)
                                            .WeightDelta(0.003083005, WeightDeltaAssertionPrecision)),
                                n => n.Id(2).ErrorGradient(0).Output(0.912)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.68595)
                                            .WeightDelta(0.003947355, WeightDeltaAssertionPrecision))))
                        .HiddenLayers(
                            l => l.Neurons(
                                n => n.Id(3).ErrorGradient(0.00236).Output(0.75568)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.71508)
                                            .WeightDelta(0.003083005, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.68595)
                                            .WeightDelta(0.003947355, WeightDeltaAssertionPrecision))
                                    .OutputSynapses(
                                        s => s.InputNeuronId(3).OutputNeuronId(4).Weight(1.00613)
                                            .WeightDelta(0.015134215, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(3).OutputNeuronId(5).Weight(0.78469)
                                            .WeightDelta(0.003464793, WeightDeltaAssertionPrecision))),
                            l => l.Neurons(
                                n => n.Id(4).ErrorGradient(0.01092).Output(0.67893)
                                    .InputSynapses(
                                        s => s.InputNeuronId(3).OutputNeuronId(4).Weight(1.00613)
                                            .WeightDelta(0.015134215, WeightDeltaAssertionPrecision))
                                    .OutputSynapses(
                                        s => s.InputNeuronId(4).OutputNeuronId(6).Weight(0.66848)
                                            .WeightDelta(0.112076637, WeightDeltaAssertionPrecision)),
                                n => n.Id(5).ErrorGradient(0.0025).Output(0.64345)
                                    .InputSynapses(
                                        s => s.InputNeuronId(3).OutputNeuronId(5).Weight(0.78469)
                                            .WeightDelta(0.003464793, WeightDeltaAssertionPrecision))
                                    .OutputSynapses(
                                        s => s.InputNeuronId(5).OutputNeuronId(6).Weight(0.22722)
                                            .WeightDelta(0.106219658, WeightDeltaAssertionPrecision))))
                        .OutputLayer(l => l
                            .Neurons(
                                n => n.Id(6).ErrorGradient(0.09001).Output(0.61197)
                                    .InputSynapses(
                                        s => s.InputNeuronId(4).OutputNeuronId(6).Weight(0.66848)
                                            .WeightDelta(0.112076637, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(5).OutputNeuronId(6).Weight(0.22722)
                                            .WeightDelta(0.106219658, WeightDeltaAssertionPrecision))))),
                    ParallelOptionsExtensions.SingleThreadedOptions())
                .QueueTrainingEpoch(e => e.Inputs(0.7122, 0.911).ExpectedOutputs(0.99).ErrorRate(0.34388)
                    .ExpectNeuralNetworkState(nn => nn
                        .InputLayer(l => l
                            .Neurons(
                                n => n.Id(1).ErrorGradient(0).Output(0.7122)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.72113)
                                            .WeightDelta(0.003539734, WeightDeltaAssertionPrecision)),
                                n => n.Id(2).ErrorGradient(0).Output(0.911)
                                    .OutputSynapses(
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.69369)
                                            .WeightDelta(0.004527798, WeightDeltaAssertionPrecision))))
                        .HiddenLayers(
                            l => l.Neurons(
                                n => n.Id(3).ErrorGradient(0.00271).Output(0.75661)
                                    .InputSynapses(
                                        s => s.InputNeuronId(1).OutputNeuronId(3).Weight(0.72113)
                                            .WeightDelta(0.003539734, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(2).OutputNeuronId(3).Weight(0.69369)
                                            .WeightDelta(0.004527798, WeightDeltaAssertionPrecision))
                                    .OutputSynapses(
                                        s => s.InputNeuronId(3).OutputNeuronId(4).Weight(1.03428)
                                            .WeightDelta(0.015832775, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(3).OutputNeuronId(5).Weight(0.79319)
                                            .WeightDelta(0.005675377, WeightDeltaAssertionPrecision))),
                            l => l.Neurons(
                                n => n.Id(4).ErrorGradient(0.01141).Output(0.68162)
                                    .InputSynapses(
                                        s => s.InputNeuronId(3).OutputNeuronId(4).Weight(1.03428)
                                            .WeightDelta(0.015832775, WeightDeltaAssertionPrecision))
                                    .OutputSynapses(
                                        s => s.InputNeuronId(4).OutputNeuronId(6).Weight(0.85801)
                                            .WeightDelta(0.098294662, WeightDeltaAssertionPrecision)),
                                n => n.Id(5).ErrorGradient(0.00409).Output(0.64421)
                                    .InputSynapses(
                                        s => s.InputNeuronId(3).OutputNeuronId(5).Weight(0.79319)
                                            .WeightDelta(0.005675377, WeightDeltaAssertionPrecision))
                                    .OutputSynapses(
                                        s => s.InputNeuronId(5).OutputNeuronId(6).Weight(0.40658)
                                            .WeightDelta(0.092899862, WeightDeltaAssertionPrecision))))
                        .OutputLayer(l => l
                            .Neurons(
                                n => n.Id(6).ErrorGradient(0.07863).Output(0.64612)
                                    .InputSynapses(
                                        s => s.InputNeuronId(4).OutputNeuronId(6).Weight(0.85801)
                                            .WeightDelta(0.098294662, WeightDeltaAssertionPrecision),
                                        s => s.InputNeuronId(5).OutputNeuronId(6).Weight(0.40658)
                                            .WeightDelta(0.092899862, WeightDeltaAssertionPrecision))))),
                    ParallelOptionsExtensions.SingleThreadedOptions())
                .PerformAllEpochs();
        }
    }
}