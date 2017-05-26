using System;

namespace NeuralNetworks.Library.NetworkInitialisation
{
    public class RandomNumberProvider : IProvideRandomNumberGeneration
    {
        private readonly Random randomNumberGenerator; 

        private RandomNumberProvider(Random randomNumberGenerator) =>
            this.randomNumberGenerator = randomNumberGenerator;

        public double GetNextRandomNumber() => randomNumberGenerator.NextDouble(); 

        public static RandomNumberProvider For(Random randomNumberGenerator)=> 
            new RandomNumberProvider(randomNumberGenerator);
    }
}
