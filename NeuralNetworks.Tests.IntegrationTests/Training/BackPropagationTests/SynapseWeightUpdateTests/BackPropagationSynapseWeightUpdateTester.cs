using System;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Library.Training.BackPropagation;
using NeuralNetworks.Tests.Support;
using NeuralNetworks.Tests.Support.Assertors;
using NeuralNetworks.Tests.Support.Builders;
using NeuralNetworks.Tests.Support.Helpers;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests.SynapseWeightUpdateTests
{
    public sealed class BackPropagationSynapseWeightUpdateTester : NeuralNetworkTester<BackPropagationSynapseWeightUpdateTester>
    {
        private static ParallelOptions UnrestrictedThreading = new ParallelOptions(); 
        private readonly IUpdateSynapseWeights synapseWeightCalculator; 

        private BackPropagationSynapseWeightUpdateTester(IUpdateSynapseWeights synapseWeightCalculator)
        {
            this.synapseWeightCalculator = synapseWeightCalculator; 
        }

        public void UpdateSynapseExpectingWeight(
            int synapseInputNeuronId, 
            int synapseOutputNeuronId, 
            double expectedWeightDelta,
            double expectedWeight)
        {
            var neuron = FindNeuronWithId(synapseOutputNeuronId);
            var synapse = FindSynapseFor(neuron, synapseInputNeuronId); 
            synapseWeightCalculator.CalculateAndUpdateInputSynapseWeights(neuron, UnrestrictedThreading); 

            var synapseAssertor = new SynapseAssertor.Builder()
                    .InputNeuronId(synapseInputNeuronId)
                    .OutputNeuronId(synapseOutputNeuronId)
                    .Weight(expectedWeight)
                    .WeightDelta(
                        expectedWeightDelta, 
                        precisionToAssertTo: targetNeuralNetwork.Context.SynapseWeightDecimalPlaces)
                    .Build(); 
            
            synapseAssertor.Assert(synapse);
        }

        private Synapse FindSynapseFor(Neuron neuron, int synapseInputNeuronId)
        { 
            var synapse = neuron.InputSynapses.First(s => s.InputNeuron.Id == synapseInputNeuronId);
            Assert.NotNull(synapse); 
            return synapse; 
        }

        public static BackPropagationSynapseWeightUpdateTester ForBackPropagation(double learningRate, double momentum)
        {
            var backPropagationSynapseUpdater = BackPropagationSynapseWeightCalculator.For(learningRate, momentum); 
            return new BackPropagationSynapseWeightUpdateTester(backPropagationSynapseUpdater);
        }

        public static BackPropagationSynapseWeightUpdateTester ForResilientBackPropagation()
        {
            var resilientBackPropagationSynapseUpdater = ResilientBackPropagationSynapseWeightCalculator.Create(); 
            return new BackPropagationSynapseWeightUpdateTester(resilientBackPropagationSynapseUpdater); 
        }
    }
}