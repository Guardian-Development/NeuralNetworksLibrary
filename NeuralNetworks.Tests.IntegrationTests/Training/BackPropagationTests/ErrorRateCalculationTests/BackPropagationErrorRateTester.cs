using System;
using System.Linq;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Library.Training.BackPropagation;
using NeuralNetworks.Tests.Support.Assertors;
using NeuralNetworks.Tests.Support.Builders;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests.ErrorRateCalculationsTests
{
    public sealed class BackPropagationErrorRateTester
    {
        private NeuralNetworkContext context; 
        private IProvideRandomNumberGeneration randomGenerator; 
        private NeuralNetwork targetNeuralNetwork; 

        private BackPropagationErrorRateTester()
        {}

        public BackPropagationErrorRateTester NeuralNetworkEnvironment(
            NeuralNetworkContext context, 
            IProvideRandomNumberGeneration randomGenerator)
        {
            this.context = context; 
            this.randomGenerator = randomGenerator;
            return this; 
        }

        public BackPropagationErrorRateTester TargetNeuralNetwork(Action<ExplicitNeuralNetworkBuilder> action)
        {
            var neuralNetworkBuilder = ExplicitNeuralNetworkBuilder.CreateForTest(context, randomGenerator); 
            action.Invoke(neuralNetworkBuilder); 
            targetNeuralNetwork = neuralNetworkBuilder.Build(); 
            return this; 
        }

        public void CalculateErrorRateForInputOrHiddenLayerNeuron(int neuronId, double expectedErrorRate)
        {
            PerformErrorRateCalculation(
                neuronId, 
                expectedErrorRate,
                (neuron, calculator) => calculator.SetNeuronErrorGradient(neuron));
        }

        public void CalculateErrorRateForOutputLayerNeuron(int neuronId, double expectedOutput, double expectedErrorRate)
        {
            PerformErrorRateCalculation(
                neuronId, 
                expectedOutput, 
                (neuron, calculator) => calculator.SetNeuronErrorGradient(neuron, expectedOutput));
        }

        private void PerformErrorRateCalculation(
            int neuronId, 
            double expectedErrorRate, 
            Action<Neuron, NeuronErrorGradientCalculator> errorRateCalculation)
        {
            var expectedErrorRateAssertor = new EqualityAssertor<double>(expectedErrorRate); 
            var errorCalculator = NeuronErrorGradientCalculator.Create();
            var targetNeuron = FindNeuronWithId(neuronId);

            errorRateCalculation(targetNeuron, errorCalculator); 
            expectedErrorRateAssertor.Assert(targetNeuron.ErrorRate);
        }

        private Neuron FindNeuronWithId(int neuronId)
        {
            var inputlayerNeurons = targetNeuralNetwork.InputLayer.Neurons; 
            var hiddenLayerNeurons = targetNeuralNetwork.HiddenLayers.SelectMany(l => l.Neurons); 
            var outputLayerNeurons = targetNeuralNetwork.OutputLayer.Neurons; 

            var allNeurons = inputlayerNeurons.Concat(hiddenLayerNeurons).Concat(outputLayerNeurons);
            var targetNeuron = allNeurons.Where(n => n.Id == neuronId).First(); 

            Assert.NotNull(targetNeuron); 
            return targetNeuron;
        }

        public static BackPropagationErrorRateTester Create()
            => new BackPropagationErrorRateTester(); 
    }
}