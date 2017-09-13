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

        public override bool Equals(object obj)
        {
            var synapse = obj as Synapse; 
            return Equals(synapse); 
        }

        private bool Equals(Synapse otherSynapse)
        {
            if(otherSynapse.InputNeuron.Id != InputNeuron.Id) return false; 
            if(otherSynapse.OutputNeuron.Id != OutputNeuron.Id) return false; 
            if(otherSynapse.Weight != Weight) return false;
            if(otherSynapse.WeightDelta != WeightDelta) return false;
            return true; 
        }

        public override int GetHashCode()
        {
            var hashCode = 37; 
            unchecked 
            {
                hashCode = (InputNeuron.Id.GetHashCode() ^ 337) * hashCode; 
                hashCode = (OutputNeuron.Id.GetHashCode() ^ 337) * hashCode; 
                hashCode = (Weight.GetHashCode() ^ 337) * hashCode;
                hashCode = (WeightDelta.GetHashCode() ^ 337) * hashCode; 
            }
            
            return hashCode; 
        }

        public static Synapse For(
            NeuralNetworkContext context,
            Neuron inputNeuron,
            Neuron outputNeuron,
            IProvideRandomNumberGeneration randomNumberGenerator)
            => new Synapse(context, inputNeuron, outputNeuron, randomNumberGenerator.GetNextRandomNumber()); 
    }
}
