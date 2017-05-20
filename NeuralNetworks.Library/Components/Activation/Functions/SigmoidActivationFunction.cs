using System;

namespace NeuralNetworks.Library.Components.Activation.Functions
{
    public sealed class SigmoidActivationFunction : IProvideNeuronActivation
    {
        private SigmoidActivationFunction() {}

        public double Activate(double sumOfWeights)
        {
            return 1.0 / (1 + Math.Exp(-1.0 * sumOfWeights));
        }

        public double Derivative(double sumOfWeights)
        {
            return sumOfWeights * (1.0 - sumOfWeights);
        }

        public static SigmoidActivationFunction Create()
        {
            return new SigmoidActivationFunction();
        }
    }
}
