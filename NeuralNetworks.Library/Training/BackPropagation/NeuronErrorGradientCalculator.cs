using System.Linq;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Extensions; 

namespace NeuralNetworks.Library.Training.BackPropagation
{
    public class NeuronErrorGradientCalculator
    {
        private readonly int errorRateDecimalPlaces;

        private NeuronErrorGradientCalculator(int errorRateDecimalPlaces)
        {
            this.errorRateDecimalPlaces = errorRateDecimalPlaces;
        }

        public void SetNeuronErrorGradient(Neuron neuron, double target)
        {
			var pureErrorRate = CalculateErrorForOutputAgainstTarget(neuron, target) *
								neuron.ActivationFunction.Derivative(neuron.Output);

            neuron.ErrorRate = pureErrorRate.RoundToDecimalPlaces(errorRateDecimalPlaces); 
        }

        public void SetNeuronErrorGradient(Neuron neuron)
        {
			var pureErrorRate = neuron.OutputSynapses.Sum(a => a.OutputNeuron.ErrorRate * a.Weight) *
							    neuron.ActivationFunction.Derivative(neuron.Output);

            neuron.ErrorRate = pureErrorRate.RoundToDecimalPlaces(errorRateDecimalPlaces); 
        }

        public double CalculateErrorForOutputAgainstTarget(Neuron neuron, double target)
        {
            return target - neuron.Output;
        }

        public static NeuronErrorGradientCalculator For(NeuralNetworkContext context)
            => new NeuronErrorGradientCalculator(context.ErrorRateDecimalPlaces); 
    }
}