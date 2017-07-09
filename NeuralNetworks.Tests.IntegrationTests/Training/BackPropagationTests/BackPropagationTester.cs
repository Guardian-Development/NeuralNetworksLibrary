using System;
using System.Collections.Generic;
using NeuralNetworks.Library;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Library.Training.BackPropagation;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class BackPropagationTester
    {
        private readonly double learningRate;
        private readonly double momentum;
        private NeuralNetworkContext context; 
        private IProvideRandomNumberGeneration randomGenerator; 
        private NeuralNetwork targetNeuralNetwork; 

        private readonly List<TrainingEpochTester<BackPropagation>> trainingStates 
            = new List<TrainingEpochTester<BackPropagation>>(); 

        private BackPropagationTester(double learningRate, double momentum)
        {
            this.momentum = momentum;
            this.learningRate = learningRate;
        }

        public BackPropagationTester NeuralNetworkEnvironment(
            NeuralNetworkContext context, 
            IProvideRandomNumberGeneration randomGenerator)
        {
            this.context = context; 
            this.randomGenerator = randomGenerator;
            return this; 
        }

        public BackPropagationTester TargetNeuralNetwork(Action<NeuralNetworkBuilder> action)
        {
            var neuralNetworkBuilder = new NeuralNetworkBuilder(context, randomGenerator); 
            action.Invoke(neuralNetworkBuilder); 
            targetNeuralNetwork = neuralNetworkBuilder.Build(); 
            return this; 
        }

        public BackPropagationTester QueueTrainingEpoch(Action<TrainingEpochTester<BackPropagation>> action)
        {
            var backPropagationTrainer = BackPropagation.WithConfiguration(targetNeuralNetwork, learningRate, momentum);
            var epochTester = TrainingEpochTester<BackPropagation>.For(backPropagationTrainer);
            trainingStates.Add(epochTester); 

            return this; 
        }

        public void PerformAllEpochs()
        {
            trainingStates.ForEach(state => state.PerformEpochAndAssert()); 
        }

        public static BackPropagationTester For(double learningRate,double momentum)
            =>  new BackPropagationTester(learningRate, momentum);
    }
}