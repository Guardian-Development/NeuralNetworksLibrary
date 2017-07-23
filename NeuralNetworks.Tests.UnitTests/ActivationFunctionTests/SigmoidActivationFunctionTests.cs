using NeuralNetworks.Library.Components.Activation.Functions;
using NeuralNetworks.Tests.Support.Helpers;
using Xunit;

namespace NeuralNetworks.Tests.UnitTests.ActivationFunctionTests
{
    public sealed class SigmoidActivationFunctionTests
    {
        [Theory]
        [InlineData(0.3775, 0.5932699921)]
        public void ActivateProducesCorrectResultPositiveValue(double activationValue, double expectedResult)
        {
            var activationFunction = SigmoidActivationFunction.Create();
            var activationResult = activationFunction.Activate(activationValue);
            DoubleAssertionHelpers.AssertWithPrecision(expectedResult, activationResult, 10);
        }

        [Theory]
        [InlineData(-0.1, 0.4750208125)]
        public void ActivateProducesCorrectResultNegativeValue(double activationValue, double expectedResult)
        {
            var activationFunction = SigmoidActivationFunction.Create();
            var activationResult = activationFunction.Activate(activationValue);
            DoubleAssertionHelpers.AssertWithPrecision(expectedResult, activationResult, 10);
        }

        [Theory]
        [InlineData(0.59, 0.229446435)]
        public void DerivativeProducesCorrectResultPositiveValue(double inputValue, double expectedResult)
        {
            var activationFunction = SigmoidActivationFunction.Create();
            var derivativeResult = activationFunction.Derivative(inputValue); 
            DoubleAssertionHelpers.AssertWithPrecision(expectedResult, derivativeResult, 10);
        }

        [Theory]
        [InlineData(-0.78, 0.2155228953)]
        public void DerviativeProducesCorrectResultNegativeValue(double inputValue, double expectedResult)
        {
            var activationFunction = SigmoidActivationFunction.Create();
            var derivativeResult = activationFunction.Derivative(inputValue);
            DoubleAssertionHelpers.AssertWithPrecision(expectedResult, derivativeResult, 10);
        }
    }
}