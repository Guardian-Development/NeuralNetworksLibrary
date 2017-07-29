using System;
using System.Linq;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Tests.Support.Builders;
using Xunit;

namespace NeuralNetworks.Tests.Support
{
    public abstract class NeuralNetworkTester<TTester> 
        where TTester : NeuralNetworkTester<TTester>
    {
        private NeuralNetworkContext context; 
        private IProvideRandomNumberGeneration randomGenerator; 
        protected NeuralNetwork targetNeuralNetwork; 

        public TTester NeuralNetworkEnvironment(
            NeuralNetworkContext context, 
            IProvideRandomNumberGeneration randomGenerator)
        {
            this.context = context; 
            this.randomGenerator = randomGenerator;
            return (TTester)this; 
        }

        public TTester TargetNeuralNetwork(Action<ExplicitNeuralNetworkBuilder> action)
        {
            var neuralNetworkBuilder = ExplicitNeuralNetworkBuilder.CreateForTest(context, randomGenerator); 
            action.Invoke(neuralNetworkBuilder); 
            targetNeuralNetwork = neuralNetworkBuilder.Build(); 
            return (TTester)this; 
        }

        protected Neuron FindNeuronWithId(int neuronId)
        {
            var inputlayerNeurons = targetNeuralNetwork.InputLayer.Neurons; 
            var hiddenLayerNeurons = targetNeuralNetwork.HiddenLayers.SelectMany(l => l.Neurons); 
            var outputLayerNeurons = targetNeuralNetwork.OutputLayer.Neurons; 

            var allNeurons = inputlayerNeurons.Concat(hiddenLayerNeurons).Concat(outputLayerNeurons);
            var targetNeuron = allNeurons.Where(n => n.Id == neuronId).First(); 

            Assert.NotNull(targetNeuron); 
            return targetNeuron;
        }
    }
}