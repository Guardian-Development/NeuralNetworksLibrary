using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components.Activation;

namespace NeuralNetworks.Library.Components
{
    public sealed class Neuron
    {
        private readonly Func<double, double> activationFunction;
        private double lastFedValue; 
        public double UserInputValue { get; set; }
        private IList<Synapse> InputConnections { get; } = new List<Synapse>();

        private Neuron(Func<double, double> activationFunction)
        {
            this.activationFunction = activationFunction;
        }

        public void AddInputConnection(Synapse connection)
        {
            InputConnections.Add(connection);
        }

        public double ActivateNeuron()
        {
            var sumOfInputValues = InputConnections.ToList()
                .Sum(synapse => synapse.Weight * synapse.Target.UserInputValue);

            lastFedValue = activationFunction.Invoke(sumOfInputValues);
            //feels like here we could use another metrics struct to hold last fed value , derivitve etc

            return lastFedValue; 
        }

        public static Neuron For(ActivationType activationType)
        {
            return new Neuron(activationType.ToFunction());
        }
    }
}
