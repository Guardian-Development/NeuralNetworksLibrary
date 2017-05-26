using System.Collections;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Library.Components.Layers
{
    public sealed class Layer: IEnumerable<Layer>
    {
        private ILogger Log => LoggerProvider.For(GetType());
        
        public List<Neuron> Neurons { get; }

        private Layer(List<Neuron> neurons)
        {
            Neurons = neurons; 
        }

        public IEnumerator<Layer> GetEnumerator()
        {
            yield return this; 
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public static Layer For(List<Neuron> neurons)
        {
            return new Layer(neurons);
        }
    }
}
