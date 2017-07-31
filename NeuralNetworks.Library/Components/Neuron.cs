using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Components.Activation.Functions;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Library.Extensions;

namespace NeuralNetworks.Library.Components
{
    public class Neuron
    {
        private static int currentNeuronId = 1;
        private static int NextNeuronId => currentNeuronId++;

        internal int Id { get; set; } = NextNeuronId; 
        
        public IProvideNeuronActivation ActivationFunction { get; }
		public List<Synapse> InputSynapses { get; } = new List<Synapse>();
		public List<Synapse> OutputSynapses { get; } = new List<Synapse>();

		public double ErrorGradient
		{
			get => roundedErrorGradient;
			set => roundedErrorGradient = value.RoundToDecimalPlaces(context.ErrorGradientDecimalPlaces);
		}

		public double Output 
        {
            get => roundedOutputValue;
            set => roundedOutputValue = value.RoundToDecimalPlaces(context.OutputDecimalPlaces); 
        }

        public double LatestFedValueFromInputSynapses { get; private set; }

		private double roundedErrorGradient;
        private double roundedOutputValue; 
		private readonly NeuralNetworkContext context;

        protected Neuron(NeuralNetworkContext context, IProvideNeuronActivation activationFunction)
        {
			this.context = context;
			ActivationFunction = activationFunction;
        }

        public virtual double CalculateOutput()
        {
            LatestFedValueFromInputSynapses = InputSynapses.Sum(a => a.Weight * a.InputNeuron.Output);
            Output = ActivationFunction.Activate(LatestFedValueFromInputSynapses);

            return Output; 
        }

        public static Neuron For(NeuralNetworkContext context, ActivationType activationType) 
            => new Neuron(context, activationType.ToNeuronActivationProvider());

        public static Neuron For(
            NeuralNetworkContext context,
            ActivationType activationType,
            IProvideRandomNumberGeneration randomNumberGeneration,
            List<Neuron> inputNeurons)
        {
            var neuron = For(context, activationType);
            ConnectNeuronWithInputNeurons(context, randomNumberGeneration, inputNeurons, neuron);
            return neuron;
        }

        private static void ConnectNeuronWithInputNeurons(
            NeuralNetworkContext context,
            IProvideRandomNumberGeneration randomNumberGeneration,
            List<Neuron> inputNeurons,
            Neuron neuron)
        {
            foreach (var inputNeuron in inputNeurons)
            {
                var synapse = Synapse.For(context, inputNeuron, neuron, randomNumberGeneration);
                inputNeuron.OutputSynapses.Add(synapse);
                neuron.InputSynapses.Add(synapse);
            }
        }
    }
}