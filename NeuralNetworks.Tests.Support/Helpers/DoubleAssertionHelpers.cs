namespace NeuralNetworks.Tests.Support.Helpers
{
    public static class DoubleAssertionHelpers
    {
        public static void AssertWithPrecision(double expected, double actual, int decimalPlaces)
        {
            Xunit.Assert.Equal(expected, actual, decimalPlaces);
        }
    }
}
