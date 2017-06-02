using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Components.Activation.Functions;
using NeuralNetworks.Library.Logging;
using NeuralNetworks.Library.NetworkInitialisation;

namespace NeuralNetworks.Library.Components
{
    public class Neuron
    {
        private static ILogger Log => LoggerProvider.For<Neuron>();

        public List<Synapse> InputSynapses { get; } = new List<Synapse>();
        public List<Synapse> OutputSynapses { get; } = new List<Synapse>();
        public double ErrorRate { get; set; }
        public double Output { get; set; }

        internal IProvideNeuronActivation ActivationFunction { get; }

        protected Neuron(IProvideNeuronActivation activationFunction)
        {
            ActivationFunction = activationFunction;
        }

        public virtual double CalculateOutput()
        {
            var inputValuesWithBias = InputSynapses.Sum(a => a.Weight * a.InputNeuron.Output);
            return Output = ActivationFunction.Activate(inputValuesWithBias);
        }

        public static Neuron For(ActivationType activationType) 
            => new Neuron(activationType.ToNeuronActivationProvider());

        public static Neuron For(
            ActivationType activationType,
            IProvideRandomNumberGeneration randomNumberGeneration,
            List<Neuron> inputNeurons)
        {
            var neuron = For(activationType);
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