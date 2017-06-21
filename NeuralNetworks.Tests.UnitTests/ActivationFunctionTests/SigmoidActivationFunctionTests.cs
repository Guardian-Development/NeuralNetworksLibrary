using NeuralNetworks.Library.Components.Activation.Functions;
using NeuralNetworks.Tests.Support.Assertors;
using Xunit;

namespace NeuralNetworks.Tests.UnitTests.ActivationFunctionTests
{
    public sealed class SigmoidActivationFunctionTests
    {
        [Theory]
        [InlineData(0.3775, 0.593269992107187)]
        public void ActivateProducesCorrectResultPositiveValue(double activationValue, double expectedResult)
        {
            var activationFunction = SigmoidActivationFunction.Create();
            var activationResult = activationFunction.Activate(activationValue);
            DoubleAssertionHelpers.AssertEqualWithinPrecision(expectedResult, activationResult);
        }

        [Theory]
        [InlineData(-0.1, 0.47502081252106)]
        public void ActivateProducesCorrectResultNegativeValue(double activationValue, double expectedResult)
        {
            var activationFunction = SigmoidActivationFunction.Create();
            var activationResult = activationFunction.Activate(activationValue);
            DoubleAssertionHelpers.AssertEqualWithinPrecision(expectedResult, activationResult);
        }
    }
}