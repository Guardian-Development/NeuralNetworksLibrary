using System.Linq;

namespace NeuralNetworks.Library.Components.Layers
{
    public abstract class Layer
    {
        public readonly Neuron[] Neurons;

        protected Layer(int neuronCount, ActivationType activationType)
        {
            Neurons = Enumerable
                .Range(0, neuronCount)
                .Select(n => Neuron.For(activationType))
                .ToArray();
        }
    }
}
