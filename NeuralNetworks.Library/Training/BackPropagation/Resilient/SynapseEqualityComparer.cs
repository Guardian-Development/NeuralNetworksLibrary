using System.Collections.Generic;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Library.Training.BackPropagation.Resilient
{
    public sealed partial class ResilientBackPropagationSynapseWeightCalculator
    {
        private class SynapseEqualityComparer : IEqualityComparer<Synapse>
        {
            public bool Equals(Synapse x, Synapse y)
            {
                if (x.InputNeuron.Id != y.InputNeuron.Id) return false;
                return x.OutputNeuron.Id == y.OutputNeuron.Id;
            }

            public int GetHashCode(Synapse obj)
            {
                var hashCode = -1254950264;
                hashCode = hashCode * -1521134295 + obj.InputNeuron.Id.GetHashCode();
                hashCode = hashCode * -1521134295 + obj.OutputNeuron.Id.GetHashCode();
                return hashCode;
            }

            public static SynapseEqualityComparer Instance() => new SynapseEqualityComparer();
        }
    }
}