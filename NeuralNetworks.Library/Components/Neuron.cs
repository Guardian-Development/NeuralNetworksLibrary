using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components.Activation;

namespace NeuralNetworks.Library.Components
{
    public sealed class Neuron
    {
        private readonly Func<double, double> activationFunction;
        private IList<Synapse> InputConnections { get; } = new List<Synapse>();

        //need to look at how best to access this + use this in math equations 
        public double Output { get; set; }

        private Neuron(Func<double, double> activationFunction)
        {
            this.activationFunction = activationFunction;
        }

        public void AddInputConnection(Synapse connection)
        {
            InputConnections.Add(connection);
        }

        public void ActivateNeuron()
        {
            var sumOfInputValues = InputConnections.ToList()
                .Sum(synapse => synapse.Weight * synapse.Target.Output);

            Output = activationFunction.Invoke(sumOfInputValues);
            //feels like here we could use another metrics struct to hold last fed value , derivitve etc
        }

        public static Neuron For(ActivationType activationType)
        {
            return new Neuron(activationType.ToFunction());
        }
    }
}
