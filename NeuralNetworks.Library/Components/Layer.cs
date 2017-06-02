using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Library.Components
{
    public abstract class Layer: IEnumerable<Layer>
    {
        private ILogger Log => LoggerProvider.For(GetType());
        
        public List<Neuron> Neurons { get; }

        protected Layer(List<Neuron> neurons)
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
    }

    public sealed class InputLayer : Layer
    {
        public List<Neuron> InputNeurons { get; }

        private InputLayer(List<Neuron> inputNeurons, List<Neuron> neuronsIncludingBias)
            : base(neuronsIncludingBias)
        {
           InputNeurons = inputNeurons;
        }

        public static InputLayer For(List<Neuron> neurons, BiasNeuron biasNeuron)
        {
            var neuronsIncludingBias = neurons.Append(biasNeuron).ToList(); 
            return new InputLayer(neurons, neuronsIncludingBias);
        }
    }

    public sealed class HiddenLayer : Layer
    {
        public HiddenLayer(List<Neuron> neuronsIncludingBias) 
            : base(neuronsIncludingBias)
        {}

        public static HiddenLayer For(List<Neuron> neurons, BiasNeuron biasNeuron)
        {
            var neuronsIncludingBias = neurons.Append(biasNeuron).ToList();
            return new HiddenLayer(neuronsIncludingBias);
        }
    }

    public sealed class OutputLayer : Layer
    {
        public OutputLayer(List<Neuron> neurons) 
            : base(neurons)
        {}

        public static OutputLayer For(List<Neuron> neurons) 
            => new OutputLayer(neurons);
    }
}
