using System;
using NeuralNetworks.Library;
using NeuralNetworks.Library.NetworkInitialisation;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class InitialNeuralNetworkBuilder
    {
        public NeuralNetwork Build() => NeuralNetwork.For(RandomNumberProvider.For(new Random(1))).Build(); 
    }
}
