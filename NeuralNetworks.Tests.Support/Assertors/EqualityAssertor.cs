using static System.FormattableString; 

namespace NeuralNetworks.Tests.Support.Assertors
{
    public class EqualityAssertor<T> : IAssert<T>
    {
        private readonly T expectedResult;

        public EqualityAssertor(T expectedResult)
        {
            this.expectedResult = expectedResult;
        }

        public void Assert(T actualItem)
        {
            Xunit.Assert.True(
                expectedResult.Equals(actualItem),
                Invariant($"Expected: {expectedResult} but Received: {actualItem}"));
        }
    }
}
