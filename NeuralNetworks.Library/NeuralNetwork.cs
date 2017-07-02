using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Library.Extensions; 

namespace NeuralNetworks.Library
{
    public sealed class NeuralNetwork
    {
        public InputLayer InputLayer { get; private set; }
        public List<HiddenLayer> HiddenLayers { get; private set; }
        public OutputLayer OutputLayer { get; private set; }

        public NeuralNetworkContext Context { get; }

        internal NeuralNetwork(NeuralNetworkContext context)
        {
            Context = context; 
        }

        internal NeuralNetwork AddInputLayer(InputLayer inputLayer)
        {
            InputLayer = inputLayer;
            return this; 
        }

        internal NeuralNetwork AddHiddenLayers(List<HiddenLayer> hiddenLayers)
        {
            HiddenLayers = hiddenLayers;
            return this; 
        }

        internal NeuralNetwork AddOutputLayer(OutputLayer outputLayer)
        {
            OutputLayer = outputLayer; 
            return this;
        }

        public double[] PredictionFor(params double[] inputs)
        {
            InputLayer.SetInputLayerOutputs(inputs);
            HiddenLayers.ApplyInReverse(layer => layer.Neurons.ForEach(a => a.CalculateOutput(Context)));
            return OutputLayer.Neurons.Select(a => a.CalculateOutput(Context)).ToArray();
        }

        public static NeuralNetworkBuilder For(
            NeuralNetworkContext context, 
            IProvideRandomNumberGeneration randomNumberGenerater = null)
        {
            randomNumberGenerater = randomNumberGenerater ?? RandomNumberProvider.For(new Random(1));
            return new NeuralNetworkBuilder(context, randomNumberGenerater);
        }
    }
}
