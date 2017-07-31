using System;
using System.Collections.Generic;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.NetworkInitialisation;

namespace NeuralNetworks.Tests.Support.Builders
{
    public class SynapseBuilder
    {
        private int inputNeuronId;
        private int outputNeuronId;
        private double weight; 
        private double weightDelta; 

        private readonly IDictionary<int, Neuron> allNeuronsInNetwork;
        private readonly IProvideRandomNumberGeneration randomGenerator;
        private readonly NeuralNetworkContext context;

        public SynapseBuilder(
            NeuralNetworkContext context,
            IDictionary<int, Neuron> allNeuronsInNetwork, 
            IProvideRandomNumberGeneration randomGenerator)
        {
            this.context = context;
            this.randomGenerator = randomGenerator;
            this.allNeuronsInNetwork = allNeuronsInNetwork;
        }

        public SynapseBuilder SynapseBetween(int inputNeuronId, int outputNeuronId, double weight)
        {
            this.inputNeuronId = inputNeuronId;
            this.outputNeuronId = outputNeuronId;
            this.weight = weight;
            return this; 
        }

        public SynapseBuilder WithWeightDelta(double weightDelta)
        {
            this.weightDelta = weightDelta; 
            return this; 
        }

        public Synapse Build()
        {
            GetNeuronsById(out var inputNeuron, out var outputNeuron);
            var synapse = Synapse.For(context, inputNeuron, outputNeuron, randomGenerator);
            ConnectNeuronsInSynapse(synapse);
            synapse.Weight = weight;
            synapse.WeightDelta = weightDelta; 

            return synapse; 
        }

        private void GetNeuronsById(out Neuron inputNeuron, out Neuron outputNeuron)
        {
            inputNeuron = GetNeuronById(inputNeuronId);
            outputNeuron = GetNeuronById(outputNeuronId);
        }

        private Neuron GetNeuronById(int id)
        {
            if(allNeuronsInNetwork.TryGetValue(id, out var neuron))
            {
                return neuron; 
            }

            throw new ArgumentException($"No Neuron exists with id: {id}");
        }

        private void ConnectNeuronsInSynapse(Synapse synapse)
        {
            synapse.InputNeuron.OutputSynapses.Add(synapse);
            synapse.OutputNeuron.InputSynapses.Add(synapse);
        }
    }
}
