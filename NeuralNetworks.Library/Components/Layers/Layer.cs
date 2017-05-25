using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Library.Components.Layers
{
    public abstract class Layer: IEnumerable<Layer>
    {
        private ILogger Log => LoggerProvider.For(GetType());
       
        public Neuron[] Neurons => null == biasNeuron ? 
            neuronsExcludingBias : neuronsExcludingBias.Append(biasNeuron).ToArray();

        public int NeuronCount => neuronsExcludingBias.Length; 

        public abstract Layer NextLayer { get; set; }

        private readonly Neuron[] neuronsExcludingBias;
        private readonly Neuron biasNeuron;

        protected Layer(int neuronCount, ActivationType activationType, Neuron bias = null)
        {
            neuronsExcludingBias = Enumerable
                .Range(0, neuronCount)
                .Select(n => Neuron.For(activationType))
                .ToArray();

            biasNeuron = bias; 
        }

        public void ActivateLayer()
        {
            Log.LogDebug("Activating Layer");
            foreach (var neuron in neuronsExcludingBias)
            {
                neuron.ActivateNeuron();
            }
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
