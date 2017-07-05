using NeuralNetworks.Tests.Support.Assertors;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Assertors
{
    public class AssertorPatternTest
    {
        [Fact]
        public void CanAssertNeuralNetworkSuccess()
        {
            var neuralNetworkAssertorBuilder = new NeuralNetworkAssertor.Builder()
                .InputLayer(l => l.Neurons(
                    n => n.Id(1).Output(1).ErrorRate(1).OutputSynapses(
                        s => s.OutputNeuronId(3).Weight(1),
                        s => s.OutputNeuronId(4).Weight(1)),
                    n => n.Id(2).Output(1).ErrorRate(1).OutputSynapses(
                        s => s.OutputNeuronId(3).Weight(1),
                        s => s.OutputNeuronId(4).Weight(1)
                    )))
                .HiddenLayers(l => l.Neurons(
                    n => n
                        .Id(3).Output(2).ErrorRate(2)
                        .OutputSynapses(
                            s => s.OutputNeuronId(5).Weight(2),
                            s => s.OutputNeuronId(6).Weight(2))
                        .InputSynapses(
                            s => s.InputNeuronId(1).Weight(1),
                            s => s.InputNeuronId(2).Weight(1)),
                    n => n
                        .Id(4).Output(2).ErrorRate(2)
                        .OutputSynapses(
                            s => s.OutputNeuronId(5).Weight(2),
                            s => s.OutputNeuronId(6).Weight(2))
                        .InputSynapses(
                            s => s.InputNeuronId(1).Weight(1),
                            s => s.InputNeuronId(2).Weight(1))))
                .OutputLayer(l => l.Neurons(
                    n => n.Id(5).Output(3).ErrorRate(3).InputSynapses(
                        s => s.InputNeuronId(3).Weight(2),
                        s => s.InputNeuronId(4).Weight(2)),
                    n => n.Id(6).Output(3).ErrorRate(3).InputSynapses(
                        s => s.InputNeuronId(3).Weight(2),
                        s => s.InputNeuronId(4).Weight(2)
                    ))).Build();
            
        }
    }
}
