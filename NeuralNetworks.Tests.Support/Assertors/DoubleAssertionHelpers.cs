using Xunit;

namespace NeuralNetworks.Tests.Support.Assertors
{
    public sealed class DoubleAssertionHelpers
    {
        public static int DoubleAssertionPrecision { get; set; } = 15;

        public static void AssertEqualWithinPrecision(double expected, double actual)
        {
            Assert.Equal(expected, actual, DoubleAssertionPrecision);
        }
    }
}
