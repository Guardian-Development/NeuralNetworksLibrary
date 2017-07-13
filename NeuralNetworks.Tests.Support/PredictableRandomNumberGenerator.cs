using NeuralNetworks.Library.NetworkInitialisation;

namespace NeuralNetworks.Tests.Support
{
    public sealed class PredictableRandomNumberGenerator : IProvideRandomNumberGeneration
    {
        private PredictableRandomNumberGenerator()
        {}
        
        public double GetNextRandomNumber() => 1.0;

        public static IProvideRandomNumberGeneration Create()
            => new PredictableRandomNumberGenerator();
    }
}
