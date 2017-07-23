using NeuralNetworks.Library.Components.Activation.Functions;
using NeuralNetworks.Tests.Support.Helpers;
using Xunit;

namespace NeuralNetworks.Tests.UnitTests.ActivationFunctionTests
{
    public sealed class TanHActivationFunctionTests
    {
        [Theory]
        [InlineData(0.3775, 0.3605343934)]
        public void ActivateProducesCorrectResultPositiveValue(double activationValue, double expectedResult)
        {
            var activationFunction = TanHActivationFunction.Create();
            var activationResult = activationFunction.Activate(activationValue);
            DoubleAssertionHelpers.AssertWithPrecision(expectedResult, activationResult, 10);
        }

        [Theory]
        [InlineData(-0.1, -0.0996679946)]
        public void ActivateProducesCorrectResultNegativeValue(double activationValue, double expectedResult)
        {
            var activationFunction = TanHActivationFunction.Create();
            var activationResult = activationFunction.Activate(activationValue);
            DoubleAssertionHelpers.AssertWithPrecision(expectedResult, activationResult, 10);
        }

        [Theory]
        [InlineData(0.36, 0.8808272706)]
        public void DerivativeProducesCorrectResultPositiveValue(double inputValue, double expectedResult)
        {
            var activationFunction = TanHActivationFunction.Create();
            var derivativeResult = activationFunction.Derivative(inputValue);
            DoubleAssertionHelpers.AssertWithPrecision(expectedResult, derivativeResult, 10);
        }

        [Theory]
        [InlineData(-0.41, 0.8490889767)]
        public void DerviativeProducesCorrectResultNegativeValue(double inputValue, double expectedResult)
        {
            var activationFunction = TanHActivationFunction.Create();
            var derivativeResult = activationFunction.Derivative(inputValue);
            DoubleAssertionHelpers.AssertWithPrecision(expectedResult, derivativeResult, 10);
        }
    }
}
