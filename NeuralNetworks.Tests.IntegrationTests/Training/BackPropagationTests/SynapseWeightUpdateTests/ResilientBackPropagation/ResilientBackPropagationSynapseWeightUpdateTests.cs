using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests.SynapseWeightUpdateTests.ResilientBackPropagation
{
    public sealed class ResilientBackPropagationSynapseWeightUpdateTests : NeuralNetworkTest
    {
        public BackPropagationSynapseWeightUpdateTester SynapseWeightUpdateTester()
            => BackPropagationSynapseWeightUpdateTester.ForResilientBackPropagation();
            
        private NeuralNetworkContext TestContext =>
            new NeuralNetworkContext(
                errorGradientDecimalPlaces: 5, 
                outputDecimalPlaces: 5, 
                synapseWeightDecimalPlaces: 5); 

        [Fact(Skip="Example test copied across from back prop test suite. Needs updating.")]
        public void CanUpdateSynapseWeightCorrectlyInputLayerNeuronToHiddenLayerNeuron()
        {
             SynapseWeightUpdateTester()
                .NeuralNetworkEnvironment(TestContext, PredictableGenerator)
                .TargetNeuralNetwork(nn => nn
                    .InputLayer(l => l
                        .Neurons(n => n.Id(1).ErrorGradient(0.15).Output(0.78).Activation(ActivationType.Sigmoid)))
                    .HiddenLayer(l => l
                        .Neurons(n => n.Id(2).ErrorGradient(-0.16).Output(0.91).Activation(ActivationType.Sigmoid)))
                    .OutputLayer(l => l 
                        .Neurons(n => n.Id(3).ErrorGradient(0.98).Output(0.77).Activation(ActivationType.Sigmoid)))
                    .Synapses(
                        s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 2, weight: 0.1423).WithWeightDelta(0.131),
                        s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 0.11)))
                .UpdateSynapseExpectingWeight(
                    synapseInputNeuronId: 1,
                    synapseOutputNeuronId: 2,
                    expectedWeightDelta : -0.0974688,
                    expectedWeight: 0.16142);
        }
    }
}