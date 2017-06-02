using System.Linq;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Library.Training.BackPropagation
{
    public static class NeuronErrorGradientOperations
    {
        public static void SetNeuronErrorGradient(Neuron neuron, double target)
        {
            neuron.ErrorRate = CalculateErrorForOutputAgainstTarget(neuron, target) *
                              neuron.ActivationFunction.Derivative(neuron.Output);
        }

        public static void SetNeuronErrorGradient(Neuron neuron)
        {
            neuron.ErrorRate = neuron.OutputSynapses.Sum(a => a.OutputNeuron.ErrorRate * a.Weight) *
                              neuron.ActivationFunction.Derivative(neuron.Output);
        }

        public static double CalculateErrorForOutputAgainstTarget(Neuron neuron, double target)
        {
            return target - neuron.Output;
        }
    }
}