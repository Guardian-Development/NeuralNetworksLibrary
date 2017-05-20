namespace NeuralNetworks.Library.Components
{
    public sealed class Synapse
    {
        private Neuron Target { get; }
        private decimal Weight { get; }

        private Synapse(Neuron target, decimal weight)
        {
            Target = target;
            Weight = weight;
        }

        public static Synapse For(Neuron target, decimal weight)
        {
            return new Synapse(target, weight);
        }
    }
}
