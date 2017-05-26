using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Components.Activation.Functions;
using NeuralNetworks.Library.Logging;
using NeuralNetworks.Library.NetworkInitialisation;

namespace NeuralNetworks.Library.Components
{
    public sealed class Neuron
    {
        private static ILogger Log => LoggerProvider.For<Neuron>();

        public List<Synapse> InputSynapses { get; } = new List<Synapse>();
        public List<Synapse> OutputSynapses { get; } = new List<Synapse>();
        public double Bias { get; set; }
        public double BiasDelta { get; set; }
        public double Gradient { get; set; }
        public double Value { get; set; }

        internal IProvideNeuronActivation ActivationFunction { get; }

        private Neuron(IProvideNeuronActivation activationFunction, double bias)
        {
            ActivationFunction = activationFunction;
            Bias = bias;
        }

        private Neuron(
            IProvideNeuronActivation activationFunction,
            IProvideRandomNumberGeneration randomNumberGeneration, 
            List<Neuron> inputNeurons,
            double bias) : this(activationFunction, bias)
        {
            foreach (var inputNeuron in inputNeurons)
            {
                var synapse = Synapse.For(inputNeuron, this, randomNumberGeneration);
                inputNeuron.OutputSynapses.Add(synapse);
                InputSynapses.Add(synapse);
            }
        }

        public void CalculateOutput()
        {
            var inputValuesWithBias = InputSynapses.Sum(a => a.Weight * a.InputNeuron.Value) + Bias;
            Value = ActivationFunction.Activate(inputValuesWithBias);
        }

        public void CalculateGradient(double target)
        {
            Gradient = CalculateError(target) * ActivationFunction.Derivative(Value);
        }

        public void CalculateGradient()
        {
            Gradient = OutputSynapses.Sum(a => a.OutputNeuron.Gradient * a.Weight) *
                       ActivationFunction.Derivative(Value);
        }

        public double CalculateError(double target)
        {
            return target - Value;
        }

        public void UpdateWeights(double learnRate, double momentum)
        {
            var prevDelta = BiasDelta;
            BiasDelta = learnRate * Gradient;
            Bias += BiasDelta + momentum * prevDelta;

            foreach (var synapse in InputSynapses)
            {
                prevDelta = synapse.WeightDelta;
                synapse.WeightDelta = learnRate * Gradient * synapse.InputNeuron.Value;
                synapse.Weight += synapse.WeightDelta + momentum * prevDelta;
            }
        }

        public static Neuron For(ActivationType activationType, double bias)
        {
            return new Neuron(activationType.ToNeuronActivationProvider(), bias);
        }

        public static Neuron For(
            ActivationType activationType,
            IProvideRandomNumberGeneration randomNumberGeneration,
            double bias,
            List<Neuron> inputNeurons)
        {
            return new Neuron(
                activationType.ToNeuronActivationProvider(), 
                randomNumberGeneration, 
                inputNeurons, 
                bias);
        }
    }
}