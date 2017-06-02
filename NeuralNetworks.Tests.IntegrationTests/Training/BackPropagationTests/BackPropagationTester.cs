using NeuralNetworks.Library;
using System;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class BackPropagationTester
    {
        private readonly double learningRate;
        private readonly double momentum;
        private NeuralNetwork targetNeuralNetwork; 

        private BackPropagationTester(double learningRate, double momentum)
        {
            this.learningRate = learningRate;
            this.momentum = momentum; 
        }

        public BackPropagationTester WithInitialNeuralNetwork(Action<InitialNeuralNetworkBuilder> actions)
        {
            var builder = new InitialNeuralNetworkBuilder();
            actions.Invoke(builder);
            targetNeuralNetwork = builder.Build();
            return this; 
        }

        public BackPropagationTester PerformTrainingEpoch(Action<TrainingEpochState> actions)
        {
            var state = new TrainingEpochState();
            actions.Invoke(state);
            state.PerformEpochAndAssertResult();
            return this; 
        }

        public static BackPropagationTester For(double learningRate, double momentum)
            => new BackPropagationTester(learningRate, momentum);
    }
}
