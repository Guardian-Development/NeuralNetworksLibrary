using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Components.Layers;

namespace NeuralNetworks.Library
{
    public sealed class NeuralNetwork
    {
        private Layer inputLayer;
        private readonly IList<Layer> hiddenLayers = new List<Layer>();
        private Layer outputLayer;

        private NeuralNetwork() {}

        public NeuralNetwork WithInputLayer(int neuronCount, ActivationType activationType)
        {
            inputLayer = InputLayer.For(neuronCount, activationType);
            return this;
        }

        public NeuralNetwork WithHiddenLayer(int neuronCount, ActivationType activationType)
        {
            var previousLayer = GetPreviousLayer();
            var hiddenLayer = HiddenLayer.For(neuronCount, activationType, previousLayer);

            hiddenLayers.Add(hiddenLayer);
            CreateNeuronConnectionsToPreviousLayer(previousLayer, hiddenLayer, 0);

            return this;
        }

        public NeuralNetwork WithOutputLayer(int neuronCount, ActivationType activationType)
        {
            var previousLayer = GetPreviousLayer();

            outputLayer = OutputLayer.For(neuronCount, activationType, previousLayer);
            CreateNeuronConnectionsToPreviousLayer(previousLayer, outputLayer, 0);

            return this;
        }

        private Layer GetPreviousLayer()
        {
            return hiddenLayers.Any() ? hiddenLayers.Last() : inputLayer;
        }

        private void CreateNeuronConnectionsToPreviousLayer(
            Layer previousLayer, Layer newLayer, decimal startingWeight)
        {
            foreach (var newLayerNeuron in newLayer.Neurons)
            {
                foreach (var previousLayerNeuron in previousLayer.Neurons)
                {
                    newLayerNeuron.AddInputConnection(Synapse.For(previousLayerNeuron, startingWeight));
                }
            }

        }

        public static NeuralNetwork Create()
        {
            return new NeuralNetwork();
        }
    }
}
