using System;

namespace NeuralNetworks.Library.Components.Activation.Functions
{
    public sealed class SigmoidActivationFunction : IProvideNeuronActivation
    {
        private SigmoidActivationFunction() {}

        public double Activate(double x) => 1.0 / (1.0 + Math.Exp(-x));

        public double Derivative(double x) 
        {
            var activationResult = Activate(x); 
            return activationResult * (1 - activationResult);
        }

        public static SigmoidActivationFunction Create() 
            => new SigmoidActivationFunction();
    }
}
