using NeuralNetworks.Library.NetworkInitialisation;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class PredictableRandomNumberGenerator : IProvideRandomNumberGeneration
    {
        public double GetNextRandomNumber() => 1.0;

        public static IProvideRandomNumberGeneration Create()
            => new PredictableRandomNumberGenerator();
    }
}
