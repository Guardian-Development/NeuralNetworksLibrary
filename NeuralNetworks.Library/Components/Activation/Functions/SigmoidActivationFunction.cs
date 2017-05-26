using System;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Library.Components.Activation.Functions
{
    public sealed class SigmoidActivationFunction : IProvideNeuronActivation
    {
        private static ILogger Log => LoggerProvider.For<SigmoidActivationFunction>();

        private SigmoidActivationFunction() {}

        public double Activate(double sumOfWeights)
        {
            var result = 1.0 / (1.0 + Math.Exp(-sumOfWeights));
            Log.LogDebug($"{nameof(Activate)} called with {nameof(sumOfWeights)} : {sumOfWeights}. Result: {result}");
            return result; 
        }

        public double Derivative(double sumOfWeights)
        {
            var result = sumOfWeights * (1 - sumOfWeights);
            Log.LogDebug($"{nameof(Derivative)} called with {nameof(sumOfWeights)}. Result: {result}");
            return result; 
        }

        public static SigmoidActivationFunction Create()
        {
            return new SigmoidActivationFunction();
        }
    }
}
