using System;

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

        public void PerformEpochAndAssertResult() => throw new NotImplementedException();
    }
}
