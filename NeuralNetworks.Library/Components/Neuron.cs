using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Components.Activation.Functions;

namespace NeuralNetworks.Library.Components
{
    public sealed class Neuron
    {
        internal IProvideNeuronActivation ActivationFunction { get; }
        internal IList<Synapse> InputConnections { get; } = new List<Synapse>();
        internal double ErrorRate { get; set; }

        //need to look at how best to access this + use this in math equations 
        public double Output { get; set; }

        public double SumOfInputValues =>
            InputConnections.ToList()
                .Sum(synapse => synapse.Weight * synapse.Source.Output);

        private Neuron(IProvideNeuronActivation activationFunction)
        {
            ActivationFunction = activationFunction;
        }

        public void AddInputConnection(Synapse connection)
        {
            InputConnections.Add(connection);
        }

        public void ActivateNeuron()
        {
            Output = ActivationFunction.Activate(SumOfInputValues);
            //feels like here we could use another metrics struct to hold last fed value , derivitve etc
        }

        public static Neuron For(ActivationType activationType)
        {
            return new Neuron(activationType.ToNeuronActivationProvider());
        }
    }
}
