using NeuralNetworks.Library.Components.Activation.Functions;
using Xunit;

namespace NeuralNetworks.Tests.UnitTests.ActivationFunctionTests
{
    public sealed class SigmoidActivationFunctionTests
    {
        [Fact]
        public void ActivateProducesCorrectResultPositiveValue()
        {
            var activationFunction = SigmoidActivationFunction.Create();
            var activationResult = activationFunction.Activate(0.3775); 
            Assert.Equal(0.593269992, activationResult);
        }
    }
}
