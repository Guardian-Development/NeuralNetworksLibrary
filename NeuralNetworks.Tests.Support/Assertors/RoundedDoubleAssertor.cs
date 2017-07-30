using NeuralNetworks.Tests.Support.Helpers;

namespace NeuralNetworks.Tests.Support.Assertors
{
    public class RoundedDoubleAssertor : IAssert<double>
    {
        private readonly double expectedResult;
        private readonly int precisionToAssertTo; 

        public RoundedDoubleAssertor(double expectedResult, int precisionToAssertTo)
        {
            this.expectedResult = expectedResult;
            this.precisionToAssertTo = precisionToAssertTo; 
        }

        public void Assert(double actualItem)
        {
            DoubleAssertionHelpers.AssertWithPrecision(
                expectedResult, 
                actualItem, 
                precisionToAssertTo); 
        }
    }
}