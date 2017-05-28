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
        public double Output { get; set; }

        internal IProvideNeuronActivation ActivationFunction { get; }

        private Neuron(IProvideNeuronActivation activationFunction, double startingBias)
        {
            ActivationFunction = activationFunction;
            Bias = startingBias;
        }

        public double CalculateOutput()
        {
            var inputValuesWithBias = InputSynapses.Sum(a => a.Weight * a.InputNeuron.Output) + Bias;
            return Output = ActivationFunction.Activate(inputValuesWithBias);
        }

        public static Neuron For(ActivationType activationType, double bias) 
            => new Neuron(activationType.ToNeuronActivationProvider(), bias);

        public static Neuron For(
            ActivationType activationType,
            IProvideRandomNumberGeneration randomNumberGeneration,
            List<Neuron> inputNeurons, 
            double bias)
        {
            var neuron = For(activationType, bias);
            ConnectNeuronWithInputNeurons(randomNumberGeneration, inputNeurons, neuron);
            return neuron; 
        }

        private static void ConnectNeuronWithInputNeurons(
            IProvideRandomNumberGeneration randomNumberGeneration, 
            List<Neuron> inputNeurons, 
            Neuron neuron)
        {
            foreach (var inputNeuron in inputNeurons)
            {
                var synapse = Synapse.For(inputNeuron, neuron, randomNumberGeneration);
                inputNeuron.OutputSynapses.Add(synapse);
                neuron.InputSynapses.Add(synapse);
            }
        }
    }
}