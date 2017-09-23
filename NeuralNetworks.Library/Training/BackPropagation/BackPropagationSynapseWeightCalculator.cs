using System.Threading.Tasks;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Extensions;

namespace NeuralNetworks.Library.Training.BackPropagation
{
    public sealed class BackPropagationSynapseWeightCalculator : IUpdateSynapseWeights
    {
        private readonly double learningRate;
        private readonly double momentum;

        private BackPropagationSynapseWeightCalculator(double learningRate, double momentum)
        {
			this.learningRate = learningRate;
			this.momentum = momentum;
        }

        public void CalculateAndUpdateInputSynapseWeights(Neuron neuron, ParallelOptions parallelOptions) 
            => neuron.InputSynapses.ParallelForEach(UpdateSynapseWeight, parallelOptions);

        private void UpdateSynapseWeight(Synapse synapse)
        {
            var prevDelta = synapse.WeightDelta;
            synapse.WeightDelta = learningRate * (synapse.OutputNeuron.ErrorGradient * synapse.InputNeuron.Output);
            synapse.Weight = synapse.Weight + synapse.WeightDelta + (momentum * prevDelta);
        }

        public static BackPropagationSynapseWeightCalculator For(double learningRate, double momentum)
            => new BackPropagationSynapseWeightCalculator(learningRate, momentum);
    }
}
