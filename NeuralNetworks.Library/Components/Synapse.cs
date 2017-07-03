using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Library.Extensions;

namespace NeuralNetworks.Library.Components
{
    public sealed class Synapse
    {
        public Neuron InputNeuron { get; }
        public Neuron OutputNeuron { get; }

        public double Weight 
        {
            get => roundedWeight;
            set => roundedWeight = value.RoundToDecimalPlaces(context.SynapseWeightDecimalPlaces);
        }

        public double WeightDelta { get; set; }

        private double roundedWeight;
        private readonly NeuralNetworkContext context;

        private Synapse(NeuralNetworkContext context, Neuron inputNeuron, Neuron outputNeuron, double weight)
        {
            this.context = context;
            InputNeuron = inputNeuron;
            OutputNeuron = outputNeuron; 
            Weight = weight;
        }

        public static Synapse For(
            NeuralNetworkContext context,
            Neuron inputNeuron,
            Neuron outputNeuron,
            IProvideRandomNumberGeneration randomNumberGenerator)
            => new Synapse(context, inputNeuron, outputNeuron, randomNumberGenerator.GetNextRandomNumber()); 
    }
}
