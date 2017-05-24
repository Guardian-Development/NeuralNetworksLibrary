using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Components.Activation.Functions;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Library.Components
{
    public sealed class Neuron
    {
        private static ILogger Log => LoggerProvider.For<Neuron>();

        public double Output { get; set; }
        public double LastCalculatedSumOfInputs { get; private set; }

        internal IProvideNeuronActivation ActivationFunction { get; }
        internal IList<Synapse> InputConnections { get; } = new List<Synapse>();
        internal double ErrorRate { get; set; }

        private double SumOfInputValues =>
            InputConnections.ToList()
                .Select(synapse => synapse.Weight * synapse.Source.Output)
                .Sum();

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
            Log.LogDebug($"Neuron activated previous {nameof(Output)} : {Output}");

            LastCalculatedSumOfInputs = SumOfInputValues;
            Output = ActivationFunction.Activate(LastCalculatedSumOfInputs);
            //TODO: looks like problem is something around here. either inputs wrong, or output not updating. 

            Log.LogDebug($"Neuron activated producing {nameof(Output)} : {Output}");
            //TODO: feels like here we could use another metrics struct to hold last fed value , derivitve etc
        }

        public static Neuron For(ActivationType activationType)
        {
            return new Neuron(activationType.ToNeuronActivationProvider());
        }
    }
}
