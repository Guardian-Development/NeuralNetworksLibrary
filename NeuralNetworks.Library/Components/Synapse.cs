namespace NeuralNetworks.Library.Components
{
    public sealed class Synapse
    {
        public Neuron Target { get; }
        public double Weight { get; set; }

        private Synapse(Neuron target, double weight)
        {
            Target = target;
            Weight = weight;
        }

        public static Synapse For(Neuron target, double weight)
        {
            return new Synapse(target, weight);
        }
    }
}
