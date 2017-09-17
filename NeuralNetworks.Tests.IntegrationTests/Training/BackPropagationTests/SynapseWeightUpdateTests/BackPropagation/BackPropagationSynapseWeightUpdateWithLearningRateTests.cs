using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests.SynapseWeightUpdateTests.BackPropagation
{               
    public sealed class BackPropagationSynapseWeightUpdateWithLearningRateTests : NeuralNetworkTest
    {
        public BackPropagationSynapseWeightUpdateTester SynapseWeightUpdateTester(double learningRate)
            => BackPropagationSynapseWeightUpdateTester.ForBackPropagation(learningRate, momentum: 0);
            
        private NeuralNetworkContext TestContext => 
            new NeuralNetworkContext(
                errorGradientDecimalPlaces: 5, 
                outputDecimalPlaces: 5, 
                synapseWeightDecimalPlaces: 5);

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyInputLayerNeuronToHiddenLayerNeuronWithLearningRate()
        {
             SynapseWeightUpdateTester(learningRate: 0.7)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.8912).Output(0.9911).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.6912).Output(0.1934).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(3).ErrorGradient(0.98).Output(0.77).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.712),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.99)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 1,
                    synapseOutputNeuronId: 2,
                    expectedWeightDelta : 0.4795333824,
                    expectedWeight: 1.19153);
            
             SynapseWeightUpdateTester(learningRate: 1.2)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.8912).Output(0.9911).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.6912).Output(0.1934).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(3).ErrorGradient(0.98).Output(0.77).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.712),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.99)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 1,
                    synapseOutputNeuronId: 2,
                    expectedWeightDelta : 0.822057984,
                    expectedWeight: 1.53406);
        }

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyInputLayerNeuronToOutputLayerNeuronWithLearningRate()
        {
            SynapseWeightUpdateTester(learningRate: 0.27231)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.23).Output(0.14).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.92).Output(0.45).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.1)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 1,
                    synapseOutputNeuronId: 2,
                    expectedWeightDelta : 0.035073528,
                    expectedWeight: 0.13507);
            
            SynapseWeightUpdateTester(learningRate: 1.8912)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.23).Output(0.14).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.92).Output(0.45).Activation(ActivationType.Sigmoid)))
                    .Synapses(s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.1)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 1,
                    synapseOutputNeuronId: 2,
                    expectedWeightDelta : 0.24358656,
                    expectedWeight: 0.34359);
        }

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyHiddenLayerNeuronToHiddenLayerNeuronWithLearningRate()
        {
             SynapseWeightUpdateTester(learningRate: 0.54129)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.2123).Output(0.333).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.1).Output(0.7451).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n =>  n.Id(3).ErrorGradient(0.7812).Output(0.8989).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(4).ErrorGradient(0.0012).Output(0.0122).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.23),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.89123),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.67123)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 2,
                    synapseOutputNeuronId: 3,
                    expectedWeightDelta : 0.315069818,
                    expectedWeight: 1.2063);
            
            SynapseWeightUpdateTester(learningRate: 2.1789)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.2123).Output(0.333).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.1).Output(0.7451).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n =>  n.Id(3).ErrorGradient(0.7812).Output(0.8989).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(4).ErrorGradient(0.0012).Output(0.0122).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.23),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.89123),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.67123)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 2,
                    synapseOutputNeuronId: 3,
                    expectedWeightDelta : 1.268276942,
                    expectedWeight: 2.15951);
        }

        [Fact]
        public void CanUpdateSynapseWeightCorrectlyHiddenLayerNeuronToOutputLayerNeuronWithLearningRate()
        {
            SynapseWeightUpdateTester(learningRate: 0.8912)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.2123).Output(0.333).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.1).Output(0.7451).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n =>  n.Id(3).ErrorGradient(0.61234).Output(0.7812).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(4).ErrorGradient(0.0012).Output(0.0122).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.23),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.89123),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.67123)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 3,
                    synapseOutputNeuronId: 4,
                    expectedWeightDelta : 0.000835447,
                    expectedWeight: 0.67207);
            
            SynapseWeightUpdateTester(learningRate: 10.12789)
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.2123).Output(0.333).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(0.1).Output(0.7451).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n =>  n.Id(3).ErrorGradient(0.61234).Output(0.7812).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(4).ErrorGradient(0.0012).Output(0.0122).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.23),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.89123),
                        s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 4, weight: 0.67123)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 3,
                    synapseOutputNeuronId: 4,
                    expectedWeightDelta : 0.009494289,
                    expectedWeight: 0.68072);
        }
    }
}