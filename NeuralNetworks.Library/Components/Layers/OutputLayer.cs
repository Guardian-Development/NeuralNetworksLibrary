using System.Linq;
using NeuralNetworks.Library.Components.Activation;

namespace NeuralNetworks.Library.Components.Layers
{
    public sealed class OutputLayer : Layer
    {
        private readonly Layer previousLayer;

        public OutputLayer(int neuronCount, ActivationType activationType, Layer previousLayer)
            : base(neuronCount, activationType)
        {
            this.previousLayer = previousLayer;
        }

        public double[] GetPrediction()
        {
            return Neurons.Select(neuron => neuron.Output).ToArray();
        }

        public static OutputLayer For(int neuronCount, ActivationType activationType, Layer previousLayer)
        {
            return new OutputLayer(neuronCount, activationType, previousLayer);
        }
    }
}
