using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Tests.Support;
using NeuralNetworks.Tests.Support.Assertors;
using NeuralNetworks.Tests.Support.Builders;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Assertors
{
    public class AssertorPatternTest
    {
        [Fact]
        public void CanAssertNeuralNetworkSuccess()
        {
            var neuralNetwork = ExplicitNeuralNetworkBuilder
                .CreateForTest(NeuralNetworkContext.MaximumPrecision, PredictableRandomNumberGenerator.Create())
                .InputLayer(l => l.Neurons(
                    n => n.Id(1).ErrorRate(1).Output(1).Activation(ActivationType.Sigmoid),
                    n => n.Id(2).ErrorRate(1).Output(1).Activation(ActivationType.Sigmoid)))
                .HiddenLayer(l => l.Neurons(
                    n => n.Id(3).ErrorRate(2).Output(2).Activation(ActivationType.Sigmoid),
                    n => n.Id(4).ErrorRate(2).Output(2).Activation(ActivationType.Sigmoid)))
                .OutputLayer(l => l.Neurons(
                    n => n.Id(5).ErrorRate(3).Output(3).Activation(ActivationType.Sigmoid),
                    n => n.Id(6).ErrorRate(3).Output(3).Activation(ActivationType.Sigmoid)))
                .Synapses(
                    s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 3, weight: 1),
                    s => s.SynapseBetween(inputNeuronId: 1, outputNeuronId: 4, weight: 1),
                    s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 3, weight: 1),
                    s => s.SynapseBetween(inputNeuronId: 2, outputNeuronId: 4, weight: 1),
                    s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 5, weight: 2),
                    s => s.SynapseBetween(inputNeuronId: 3, outputNeuronId: 6, weight: 2),
                    s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 5, weight: 2),
                    s => s.SynapseBetween(inputNeuronId: 4, outputNeuronId: 6, weight: 2))
                .Build();


            var neuralNetworkAssertor = new NeuralNetworkAssertor.Builder()
                .InputLayer(l => l.Neurons(
                    n => n.Id(1).Output(1).ErrorRate(1).OutputSynapses(
                        s => s.OutputNeuronId(3).Weight(1),
                        s => s.OutputNeuronId(4).Weight(1)),
                    n => n.Id(2).Output(1).ErrorRate(1).OutputSynapses(
                        s => s.OutputNeuronId(3).Weight(1),
                        s => s.OutputNeuronId(4).Weight(1))))
                .HiddenLayers(l => l.Neurons(
                    n => n
                        .Id(3).Output(2).ErrorRate(2)
                        .InputSynapses(
                            s => s.InputNeuronId(1).Weight(1),
                            s => s.InputNeuronId(2).Weight(1))
                        .OutputSynapses(
                            s => s.OutputNeuronId(5).Weight(2),
                            s => s.OutputNeuronId(6).Weight(2)),
                    n => n
                        .Id(4).Output(2).ErrorRate(2)
                        .InputSynapses(
                            s => s.InputNeuronId(1).Weight(1),
                            s => s.InputNeuronId(2).Weight(1))
                        .OutputSynapses(
                            s => s.OutputNeuronId(5).Weight(2),
                            s => s.OutputNeuronId(6).Weight(2))))
                .OutputLayer(l => l.Neurons(
                    n => n.Id(5).Output(3).ErrorRate(3).InputSynapses(
                        s => s.InputNeuronId(3).Weight(2),
                        s => s.InputNeuronId(4).Weight(2)),
                    n => n.Id(6).Output(3).ErrorRate(3).InputSynapses(
                        s => s.InputNeuronId(3).Weight(2),
                        s => s.InputNeuronId(4).Weight(2))))
                .Build();

            neuralNetworkAssertor.Assert(neuralNetwork);
        }
    }
}
