using System.Linq;
using NeuralNetworks.Library.Components.Activation;

namespace NeuralNetworks.Library.Components.Layers
{
    public abstract class Layer
    {
        public Neuron[] Neurons { get; }

        protected Layer(int neuronCount, ActivationType activationType)
        {
            Neurons = Enumerable
                .Range(0, neuronCount)
                .Select(n => Neuron.For(activationType))
                .ToArray();
        }
    }
}
