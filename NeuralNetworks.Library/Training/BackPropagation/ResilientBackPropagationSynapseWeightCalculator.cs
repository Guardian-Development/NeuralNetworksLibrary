using System.Threading.Tasks;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Library.Training.BackPropagation
{
    public sealed class ResilientBackPropagationSynapseWeightCalculator : IUpdateSynapseWeights
    {
        private ResilientBackPropagationSynapseWeightCalculator()
        {}
        
        public void CalculateAndUpdateInputSynapseWeights(Neuron neuron, ParallelOptions parallelOptions)
        {
            throw new System.NotImplementedException();
        }

        public static ResilientBackPropagationSynapseWeightCalculator Create()
            => new ResilientBackPropagationSynapseWeightCalculator(); 
    }
}
