using System;
using System.Collections.Generic;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Data;
using NeuralNetworks.Library.Training;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class TrainingEpochState
    {
        private double[] inputs;
        private double[] expectedOutputs;
        private double expectedErrorRate;
        private NeuralNetworkAssertor neuralNetworkAssertor;

        public TrainingEpochState Inputs(params double[] givenInputs)
        {
            inputs = givenInputs;
            return this;
        }

        public TrainingEpochState ExpectedOutputs(params double[] outputs)
        {
            expectedOutputs = outputs;
            return this;
        }

        public TrainingEpochState ExpectedErrorRate(double errorRate)
        {
            expectedErrorRate = errorRate;
            return this;
        }

        public TrainingEpochState ExpectNeuralNetworkState(Action<NeuralNetworkAssertorBuilder> actions)
        {
            var builder = new NeuralNetworkAssertorBuilder();
            actions.Invoke(builder);
            neuralNetworkAssertor = builder.Build();
            return this;
        }

        public void PerformEpochAndAssertResult(
            ITrainNeuralNetworks trainer, 
            List<(int id, Neuron neuron)> targetNeuralNetworkNeurons,
            List<Synapse> targetNeuralNetworkSynapses)
        {
            var trainingData = TrainingDataSet.For(inputs, expectedOutputs); 

            var errorRate = trainer.PerformSingleEpochProducingErrorRate(trainingData);

            neuralNetworkAssertor.Assert(targetNeuralNetworkNeurons, targetNeuralNetworkSynapses);
            Assert.Equal(expectedErrorRate, errorRate);
        }
    }
}