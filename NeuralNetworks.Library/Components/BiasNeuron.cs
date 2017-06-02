using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Components.Activation.Functions;

namespace NeuralNetworks.Library.Components
{
    public class BiasNeuron : Neuron
    {
        public BiasNeuron(IProvideNeuronActivation activationFunction, double constantOutput)
            : base(activationFunction)
        {
            Output = constantOutput; 
        }

        public override double CalculateOutput() => Output;

        public static BiasNeuron For(ActivationType activationType, double constantOutput) 
            => new BiasNeuron(activationType.ToNeuronActivationProvider(), constantOutput);
    }
}