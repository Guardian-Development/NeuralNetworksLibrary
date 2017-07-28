using System;
using NeuralNetworks.Library;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Library.Training.BackPropagation;
using NeuralNetworks.Tests.Support.Builders;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests.SynapseWeightUpdateTests
{
    public sealed class BackPropagationSynapseWeightUpdateTester
    {
        private NeuralNetworkContext context; 
        private IProvideRandomNumberGeneration randomGenerator; 
        private NeuralNetwork targetNeuralNetwork; 

        private readonly SynapseWeightCalculator synapseWeightCalculator; 

        private BackPropagationSynapseWeightUpdateTester(double learningRate, double momentum)
        {
            synapseWeightCalculator = SynapseWeightCalculator.For(learningRate, momentum); 
        }

        public BackPropagationSynapseWeightUpdateTester NeuralNetworkEnvironment(
            NeuralNetworkContext context, 
            IProvideRandomNumberGeneration randomGenerator)
        {
            this.context = context; 
            this.randomGenerator = randomGenerator;
            return this; 
        }

        public BackPropagationSynapseWeightUpdateTester TargetNeuralNetwork(Action<ExplicitNeuralNetworkBuilder> action)
        {
            var neuralNetworkBuilder = ExplicitNeuralNetworkBuilder.CreateForTest(context, randomGenerator); 
            action.Invoke(neuralNetworkBuilder); 
            targetNeuralNetwork = neuralNetworkBuilder.Build(); 
            return this; 
        }

        public void UpdateSynapseExpectingWeight(
            int synapseInputNeuronId, 
            int synapseOutputNeuronId, 
            double expectedWeight)
        {
            throw new NotImplementedException();
        }

        public static BackPropagationSynapseWeightUpdateTester Create(double learningRate, double momentum)
            => new BackPropagationSynapseWeightUpdateTester(learningRate, momentum); 
    }
}