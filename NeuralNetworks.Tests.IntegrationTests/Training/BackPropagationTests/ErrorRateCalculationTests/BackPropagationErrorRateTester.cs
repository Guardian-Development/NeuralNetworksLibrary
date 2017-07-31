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

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests.ErrorGradientCalculationsTests
{
    public sealed class BackPropagationErrorGradientTester : NeuralNetworkTester<BackPropagationErrorGradientTester>
    {

        private BackPropagationErrorGradientTester()
        {}

        public BackPropagationErrorGradientTester ActivateNeuronWithId(int neuronId)
        {
            var targetNeuron = FindNeuronWithId(neuronId);
            targetNeuron.CalculateOutput();
            return this;
        }

        public void CalculateErrorGradientForHiddenLayerNeuron(int neuronId, double errorGradient)
        {
            PerformErrorGradientCalculation(
                neuronId, 
                errorGradient,
                (neuron, calculator) => calculator.SetNeuronErrorGradient(neuron));
        }

        public void CalculateErrorGradientForOutputLayerNeuron(int neuronId, double expectedOutput, double errorGradient)
        {
            PerformErrorGradientCalculation(
                neuronId, 
                errorGradient, 
                (neuron, calculator) => calculator.SetNeuronErrorGradient(neuron, expectedOutput));
        }

        private void PerformErrorGradientCalculation(
            int neuronId, 
            double errorGradient, 
            Action<Neuron, NeuronErrorGradientCalculator> errorGradientCalculation)
        {
            var expectedErrorGradientAssertor = new EqualityAssertor<double>(errorGradient); 
            var errorCalculator = NeuronErrorGradientCalculator.Create();
            var targetNeuron = FindNeuronWithId(neuronId);

            errorGradientCalculation(targetNeuron, errorCalculator); 
            expectedErrorGradientAssertor.Assert(targetNeuron.ErrorGradient);
        }

        public static BackPropagationErrorGradientTester Create()
            => new BackPropagationErrorGradientTester(); 
    }
}