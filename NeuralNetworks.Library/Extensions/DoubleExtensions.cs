using System;

namespace NeuralNetworks.Library.Extensions
{
    public static class DoubleExtensions
    {
        public static double RoundToDecimalPlaces(this double source, int decimalPlaces)
        {
            return Math.Round(source, decimalPlaces);
		}
    }
}
