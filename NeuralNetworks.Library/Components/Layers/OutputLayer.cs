using System;
using System.Linq;
using NeuralNetworks.Library.Components.Activation;

namespace NeuralNetworks.Library.Components.Layers
{
    public sealed class OutputLayer : Layer
    {
        public OutputLayer(int neuronCount, ActivationType activationType)
            : base(neuronCount, activationType)
        {}

        public double[] GetPrediction()
        {
            return Neurons.Select(neuron => neuron.Output).ToArray();
        }

        public static OutputLayer For(int neuronCount, ActivationType activationType)
        {
            return new OutputLayer(neuronCount, activationType);
        }

        public override Layer NextLayer
        {
            get => throw new InvalidOperationException(
                $"{nameof(NextLayer)} should not be called on the output layer");
            set => throw new InvalidOperationException(
                $"{nameof(NextLayer)} should not be called on the output layer");
        }
    }
}
