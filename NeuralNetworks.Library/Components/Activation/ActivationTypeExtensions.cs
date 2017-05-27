using System;
using System.Collections.Generic;
using NeuralNetworks.Library.Components.Activation.Functions;

namespace NeuralNetworks.Library.Components.Activation
{
    public static class ActivationTypeExtensions
    {
        private static readonly IDictionary<ActivationType, IProvideNeuronActivation>
            ActivationFunctions = new Dictionary<ActivationType, IProvideNeuronActivation>
            {
                {ActivationType.Sigmoid, SigmoidActivationFunction.Create()},
                {ActivationType.TanH, TanHActivationFunction.Create()}
            };

        public static IProvideNeuronActivation ToNeuronActivationProvider(
            this ActivationType activationType)
        {
            ActivationFunctions.TryGetValue(activationType, out var activationFunction);
            if (activationFunction == null)
            {
                throw new NotSupportedException(
                    $"The activaion type {nameof(activationType)} does not have a corresponding {nameof(IProvideNeuronActivation)} function.");
            }

            return activationFunction; 
        }
    }
}
