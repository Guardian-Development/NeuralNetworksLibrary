using NeuralNetworks.Library.NetworkInitialisation;

namespace NeuralNetworks.Library.Components
{
    public sealed class Synapse
    {
        public Neuron InputNeuron { get; }
        public Neuron OutputNeuron { get; }
        public double Weight { get; set; }
        public double WeightDelta { get; set; }

        private Synapse(Neuron inputNeuron, Neuron outputNeuron, double weight)
        {
            InputNeuron = inputNeuron;
            OutputNeuron = outputNeuron; 
            Weight = weight;
        }

        public static Synapse For(
            Neuron inputNeuron, 
            Neuron outputNeuron, 
            IProvideRandomNumberGeneration randomNumberGenerator)
        {
            return new Synapse(inputNeuron, outputNeuron, randomNumberGenerator.GetNextRandomNumber());
        }
    }
}
