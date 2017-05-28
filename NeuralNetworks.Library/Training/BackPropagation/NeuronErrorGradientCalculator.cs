using System.Linq;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Library.Training.BackPropagation
{
    public static class NeuronErrorGradientCalculator
    {
        public static void SetNeuronErrorGradient(Neuron neuron, double target)
        {
            neuron.Gradient = CalculateErrorForOutputAgainstTarget(neuron, target) *
                              neuron.ActivationFunction.Derivative(neuron.Output);
        }

        public static void SetNeuronErrorGradient(Neuron neuron)
        {
            neuron.Gradient = neuron.OutputSynapses.Sum(a => a.OutputNeuron.Gradient * a.Weight) *
                              neuron.ActivationFunction.Derivative(neuron.Output);
        }

        public static double CalculateErrorForOutputAgainstTarget(Neuron neuron, double target)
        {
            return target - neuron.Output;
        }
    }
}