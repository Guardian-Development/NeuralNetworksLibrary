using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests.SynapseWeightUpdateTests.BackPropagation
{               
    public sealed class BackPropagationSynapseWeightUpdateWithMomentumTests : NeuralNetworkTest
    {
        public BackPropagationSynapseWeightUpdateTester SynapseWeightUpdateTester(double momentum)
            => BackPropagationSynapseWeightUpdateTester.ForBackPropagation(learningRate: 1, momentum: momentum);
            
        private NeuralNetworkContext TestContext => 
            new NeuralNetworkContext(
                errorGradientDecimalPlaces: 5, 
                outputDecimalPlaces: 5, 
                synapseWeightDecimalPlaces: 5);

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyInputLayerNeuronToHiddenLayerNeuronWithMomentum()
        {
             SynapseWeightUpdateTester(momentum: 0.634)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.7145).Output(0.8989).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.264).Output(0.2129).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(3).ErrorGradient(0.671).Output(0.9123).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.8914).WithWeightDelta(0.125),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.99)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 1,
                    synapseOutputNeuronId: 2,
                    expectedWeightDelta : 0.2373096,
                    expectedWeight: 1.20796);
            
            SynapseWeightUpdateTester(momentum: 1.234)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.7145).Output(0.8989).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.264).Output(0.2129).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(3).ErrorGradient(0.671).Output(0.9123).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.8914).WithWeightDelta(0.125),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.99)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 1,
                    synapseOutputNeuronId: 2,
                    expectedWeightDelta : 0.2373096,
                    expectedWeight: 1.28296);
        }

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyInputLayerNeuronToOutputLayerNeuronWithMomentum()
        {
            SynapseWeightUpdateTester(momentum: 0.1289)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.121).Output(0.4545).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.897).Output(0.243).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.1).WithWeightDelta(0.231)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 1,
                    synapseOutputNeuronId: 2,
                    expectedWeightDelta : 0.4076865,
                    expectedWeight: 0.53746);
            
            SynapseWeightUpdateTester(momentum: 1.1845)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.121).Output(0.4545).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.897).Output(0.243).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.1).WithWeightDelta(0.231)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 1,
                    synapseOutputNeuronId: 2,
                    expectedWeightDelta : 0.4076865,
                    expectedWeight: 0.78131);
        }

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyHiddenLayerNeuronToHiddenLayerNeuronWithMomentum()
        {
             SynapseWeightUpdateTester(momentum: 0.54129)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.2123).Output(0.333).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.891).Output(0.873).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n =>  n.Id(3).ErrorGradient(0.5612).Output(0.6321).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(4).ErrorGradient(0.0012).Output(0.0122).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.129),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.8165).WithWeightDelta(0.781),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.576)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 2,
                    synapseOutputNeuronId: 3,
                    expectedWeightDelta : 0.4899276,
                    expectedWeight: 1.72918);
            
            SynapseWeightUpdateTester(momentum: 1.9823)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.2123).Output(0.333).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.891).Output(0.873).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n =>  n.Id(3).ErrorGradient(0.5612).Output(0.6321).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(4).ErrorGradient(0.0012).Output(0.0122).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.129),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.8165).WithWeightDelta(0.781),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.576)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 2,
                    synapseOutputNeuronId: 3,
                    expectedWeightDelta : 0.4899276,
                    expectedWeight: 2.8546);
        }

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyHiddenLayerNeuronToOutputLayerNeuronWithMomentum()
        {
            SynapseWeightUpdateTester(momentum: 0.4218)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.2123).Output(0.333).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.1).Output(0.7451).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n =>  n.Id(3).ErrorGradient(0.5123).Output(0.4123).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(4).ErrorGradient(0.8953).Output(0.2323).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.23),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.89123),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.2315).WithWeightDelta(0.243)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 3,
                    synapseOutputNeuronId: 4,
                    expectedWeightDelta : 0.36913219,
                    expectedWeight: 0.70313);
            
            SynapseWeightUpdateTester(momentum: 11.72)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.2123).Output(0.333).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.1).Output(0.7451).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n =>  n.Id(3).ErrorGradient(0.5123).Output(0.4123).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(4).ErrorGradient(0.8953).Output(0.2323).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.23),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.89123),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.2315).WithWeightDelta(0.243)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 3,
                    synapseOutputNeuronId: 4,
                    expectedWeightDelta : 0.36913219,
                    expectedWeight: 3.44859);
        }
    }
}