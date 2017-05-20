using System;
using System.Collections.Generic;
using NeuralNetworks.Library.Components.Activation;

namespace NeuralNetworks.Library.Components
{
    public sealed class Neuron
    {
        private readonly Func<double, double> activationFunction;
        private IList<Synapse> InputConnections { get; } = new List<Synapse>();

        private Neuron(Func<double, double> activationFunction)
        {
            this.activationFunction = activationFunction;
        }

        public void AddInputConnection(Synapse connection)
        {
            InputConnections.Add(connection);
        }

        public static Neuron For(ActivationType activationType)
        {
            return new Neuron(activationType.ToFunction());
        }
    }
}
