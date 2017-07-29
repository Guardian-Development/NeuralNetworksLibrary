using System;

namespace NeuralNetworks.Library.Components.Activation.Functions
{
    public sealed class TanHActivationFunction : IProvideNeuronActivation
    {
        private TanHActivationFunction() {}

        public double Activate(double x) => Math.Tanh(x);

        public double Derivative(double x) => 1 - Math.Pow(Math.Tanh(x), 2);

        public static TanHActivationFunction Create() => 
            new TanHActivationFunction();
    }
}
