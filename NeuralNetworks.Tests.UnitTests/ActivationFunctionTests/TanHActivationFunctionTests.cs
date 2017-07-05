using NeuralNetworks.Library.Components.Activation.Functions;
using NeuralNetworks.Tests.Support.Helpers;
using Xunit;

namespace NeuralNetworks.Tests.UnitTests.ActivationFunctionTests
{
    public sealed class TanHActivationFunctionTests
    {
        [Theory]
        [InlineData(0.3775, 0.36053439339729690393031538636976)]
        public void ActivateProducesCorrectResultPositiveValue(double activationValue, double expectedResult)
        {
            var activationFunction = TanHActivationFunction.Create();
            var activationResult = activationFunction.Activate(activationValue);
            DoubleAssertionHelpers.AssertWithPrecision(expectedResult, activationResult, 15);
        }

        [Theory]
        [InlineData(-0.1, -0.09966799462495581711830508367835)]
        public void ActivateProducesCorrectResultNegativeValue(double activationValue, double expectedResult)
        {
            var activationFunction = TanHActivationFunction.Create();
            var activationResult = activationFunction.Activate(activationValue);
            DoubleAssertionHelpers.AssertWithPrecision(expectedResult, activationResult, 15);
        }

        [Theory]
        [InlineData(0.36, 0.88082727063587941532668799659177)]
        public void DerivativeProducesCorrectResultPositiveValue(double inputValue, double expectedResult)
        {
            var activationFunction = TanHActivationFunction.Create();
            var derivativeResult = activationFunction.Derivative(inputValue);
            DoubleAssertionHelpers.AssertWithPrecision(expectedResult, derivativeResult, 15);
        }

        [Theory]
        [InlineData(-0.41, 0.84908897672574999550376869771353)]
        public void DerviativeProducesCorrectResultNegativeValue(double inputValue, double expectedResult)
        {
            var activationFunction = TanHActivationFunction.Create();
            var derivativeResult = activationFunction.Derivative(inputValue);
            DoubleAssertionHelpers.AssertWithPrecision(expectedResult, derivativeResult, 15);
        }
    }
}
