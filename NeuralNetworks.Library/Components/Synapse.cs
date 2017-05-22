namespace NeuralNetworks.Library.Components
{
    public sealed class Synapse
    {
        public Neuron Source { get; }
        public double Weight { get; set; }

        private Synapse(Neuron source, double weight)
        {
            Source = source;
            Weight = weight;
        }

        public static Synapse For(Neuron source, double weight)
        {
            return new Synapse(source, weight);
        }
    }
}
