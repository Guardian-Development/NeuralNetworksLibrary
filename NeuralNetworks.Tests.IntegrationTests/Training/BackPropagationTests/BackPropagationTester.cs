using NeuralNetworks.Library;
using System;
using System.Collections.Generic;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Training.BackPropagation;
using NeuralNetworks.Tests.Support.Builders;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class BackPropagationTester
    {
        private readonly double learningRate;
        private readonly double momentum;
        private NeuralNetwork targetNeuralNetwork;
        private List<(int id, Neuron neuron)> targetNeuralNetworkNeuronsWithId;
        private List<Synapse> targetNeuralNetworkSynapses;

        private BackPropagationTester(double learningRate, double momentum)
        {
            this.learningRate = learningRate;
            this.momentum = momentum;
        }

        public BackPropagationTester WithTargetNeuralNetwork(Action<InitialNeuralNetworkBuilder> actions)
        {
            var builder = new InitialNeuralNetworkBuilder();
            actions.Invoke(builder);
            (targetNeuralNetwork, targetNeuralNetworkNeuronsWithId, targetNeuralNetworkSynapses) = builder.Build();
            return this;
        }

        public BackPropagationTester PerformTrainingEpoch(Action<TrainingEpochState> actions)
        {
            var state = new TrainingEpochState();
            actions.Invoke(state);

            var trainer = BackPropagation.WithConfiguration(targetNeuralNetwork, learningRate, momentum);
            state.PerformEpochAndAssertResult(trainer, targetNeuralNetworkNeuronsWithId, targetNeuralNetworkSynapses);
            return this;
        }

        public static BackPropagationTester For(double learningRate, double momentum)
            => new BackPropagationTester(learningRate, momentum);
    }
}