using System.Collections;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components.Activation;

namespace NeuralNetworks.Library.Components.Layers
{
    public abstract class Layer: IEnumerable<Layer>
    {
        public Neuron[] Neurons { get; }
        public abstract Layer NextLayer { get; set; }

        public void ActivateLayer()
        {
            foreach (var neuron in Neurons)
            {
                neuron.ActivateNeuron(); 
            }
        }

        protected Layer(int neuronCount, ActivationType activationType)
        {
            Neurons = Enumerable
                .Range(0, neuronCount)
                .Select(n => Neuron.For(activationType))
                .ToArray();
        }

        public IEnumerator<Layer> GetEnumerator()
        {
            yield return this; 
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
