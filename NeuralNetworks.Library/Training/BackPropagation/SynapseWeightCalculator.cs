using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Library.Training.BackPropagation
{
    public class SynapseWeightCalculator
    {
        private readonly double learningRate;
        private readonly double momentum;

        private SynapseWeightCalculator(double learningRate, double momentum)
        {
			this.learningRate = learningRate;
			this.momentum = momentum;
        }

        public void CalculateAndUpdateInputSynapseWeights(Neuron neuron) 
            => neuron.InputSynapses.ForEach(UpdateSynapseWeight);

        private void UpdateSynapseWeight(Synapse synapse)
        {
            var prevDelta = synapse.WeightDelta;
            synapse.WeightDelta = learningRate * synapse.OutputNeuron.ErrorRate * synapse.InputNeuron.Output;
            synapse.Weight = synapse.Weight + synapse.WeightDelta + momentum * prevDelta;
        }

        public static SynapseWeightCalculator For(double learningRate, double momentum)
            => new SynapseWeightCalculator(learningRate, momentum);
    }
}
