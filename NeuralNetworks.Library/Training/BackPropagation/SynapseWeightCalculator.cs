using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Extensions; 

namespace NeuralNetworks.Library.Training.BackPropagation
{
    public class SynapseWeightCalculator
    {
        private readonly int synapseWeightDecimalPlaces;
        private readonly double learningRate;
        private readonly double momentum;

        private SynapseWeightCalculator(
            int synapseWeightDecimalPlaces, 
            double learningRate, 
            double momentum)
        {
            this.synapseWeightDecimalPlaces = synapseWeightDecimalPlaces;
			this.learningRate = learningRate;
			this.momentum = momentum;
        }

        public void CalculateAndUpdateInputSynapseWeights(Neuron neuron) 
            => neuron.InputSynapses.ForEach(UpdateSynapseWeight);

        private void UpdateSynapseWeight(Synapse synapse)
        {
            var prevDelta = synapse.WeightDelta;
            synapse.WeightDelta = learningRate * synapse.OutputNeuron.ErrorRate * synapse.InputNeuron.Output;
            var pureSynapseWeight = synapse.Weight + synapse.WeightDelta + momentum * prevDelta;

            synapse.Weight = pureSynapseWeight.RoundToDecimalPlaces(synapseWeightDecimalPlaces); 
        }

        public static SynapseWeightCalculator For(
            NeuralNetworkContext context, 
            double learningRate, 
            double momentum)
        {
            return new SynapseWeightCalculator(context.SynapseWeightDecimalPlaces, learningRate, momentum);
        }
    }
}
