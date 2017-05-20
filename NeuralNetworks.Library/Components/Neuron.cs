using System.Collections.Generic;

namespace NeuralNetworks.Library.Components
{
    public sealed class Neuron
    {
        private readonly ActivationType activationType;
        private IList<Synapse> InputConnections { get; } = new List<Synapse>();

        private Neuron(ActivationType activationType)
        {
            this.activationType = activationType;
        }

        public void AddInputConnection(Synapse connection)
        {
            InputConnections.Add(connection);
        }

        public static Neuron For(ActivationType activationType)
        {
            return new Neuron(activationType);
        }
    }
}
