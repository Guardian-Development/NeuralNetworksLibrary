using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Components.Activation.Functions;

namespace NeuralNetworks.Library.Components
{
    public sealed class BiasNeuron : Neuron
    {
        public BiasNeuron(
            NeuralNetworkContext context, 
            IProvideNeuronActivation activationFunction, 
            double constantOutput)
            : base(context, activationFunction)
        {
            Output = constantOutput; 
        }

        public override double CalculateOutput() => Output;

        public static BiasNeuron For(
            NeuralNetworkContext context, 
            ActivationType activationType, 
            double constantOutput) 
            => new BiasNeuron(context, activationType.ToNeuronActivationProvider(), constantOutput);
    }
}