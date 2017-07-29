using System;
using System.Linq;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Library.Training.BackPropagation;
using NeuralNetworks.Tests.Support;
using NeuralNetworks.Tests.Support.Assertors;
using NeuralNetworks.Tests.Support.Builders;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests.ErrorRateCalculationsTests
{
    public sealed class BackPropagationErrorRateTester : NeuralNetworkTester<BackPropagationErrorRateTester>
    {

        private BackPropagationErrorRateTester()
        {}

        public BackPropagationErrorRateTester ActivateNeuronWithId(int neuronId)
        {
            var targetNeuron = FindNeuronWithId(neuronId);
            targetNeuron.CalculateOutput();
            return this;
        }

        public void CalculateErrorRateForHiddenLayerNeuron(int neuronId, double errorRate)
        {
            PerformErrorRateCalculation(
                neuronId, 
                errorRate,
                (neuron, calculator) => calculator.SetNeuronErrorGradient(neuron));
        }

        public void CalculateErrorRateForOutputLayerNeuron(int neuronId, double expectedOutput, double errorRate)
        {
            PerformErrorRateCalculation(
                neuronId, 
                errorRate, 
                (neuron, calculator) => calculator.SetNeuronErrorGradient(neuron, expectedOutput));
        }

        private void PerformErrorRateCalculation(
            int neuronId, 
            double errorRate, 
            Action<Neuron, NeuronErrorGradientCalculator> errorRateCalculation)
        {
            var expectedErrorRateAssertor = new EqualityAssertor<double>(errorRate); 
            var errorCalculator = NeuronErrorGradientCalculator.Create();
            var targetNeuron = FindNeuronWithId(neuronId);

            errorRateCalculation(targetNeuron, errorCalculator); 
            expectedErrorRateAssertor.Assert(targetNeuron.ErrorRate);
        }

        public static BackPropagationErrorRateTester Create()
            => new BackPropagationErrorRateTester(); 
    }
}