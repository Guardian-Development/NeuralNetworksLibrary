using System.Threading.Tasks;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Library.Training.BackPropagation
{
    public interface IUpdateSynapseWeights
    {
        void CalculateAndUpdateInputSynapseWeights(Neuron neuron, ParallelOptions parallelOptions); 
    }
}