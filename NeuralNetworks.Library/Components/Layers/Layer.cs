using System.Collections;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Library.Components.Layers
{
    public sealed class Layer: IEnumerable<Layer>
    {
        private ILogger Log => LoggerProvider.For(GetType());
        
        public IEnumerable<Neuron> Neurons { get; }

        private Layer(IEnumerable<Neuron> neurons)
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

        public static Layer For(IEnumerable<Neuron> neurons)
        {
            return new Layer(neurons);
        }
    }
}
