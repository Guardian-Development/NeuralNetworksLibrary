using System;

namespace NeuralNetworks.Library.NetworkInitialisation
{
    public sealed class RandomNumberProvider : IProvideRandomNumberGeneration
    {
        private readonly Random randomNumberGenerator; 

        private RandomNumberProvider(Random randomGenerator)
        {
			randomNumberGenerator = randomGenerator;
        }

        public double GetNextRandomNumber() => randomNumberGenerator.NextDouble(); 

        public static RandomNumberProvider For(Random randomGenerator) => 
            new RandomNumberProvider(randomGenerator);
    }
}
