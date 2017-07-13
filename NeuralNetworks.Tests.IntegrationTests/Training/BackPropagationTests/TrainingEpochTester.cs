using System;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Data;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Library.Training;
using NeuralNetworks.Tests.Support.Assertors;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class TrainingEpochTester<TTrainingMethod>
        where TTrainingMethod : NeuralNetworkTrainer
    {
        private double[] inputs; 
        private double[] outputs; 
        private double errorRate; 
        private IAssert<NeuralNetwork> neuralNetworkAssertor;
        private readonly TTrainingMethod trainingMethod; 

        private TrainingEpochTester(TTrainingMethod trainingMethod)
        {
            this.trainingMethod = trainingMethod; 
        }

        public TrainingEpochTester<TTrainingMethod> Inputs(params double[] inputs)
        {
            this.inputs = inputs; 
            return this; 
        }

        public TrainingEpochTester<TTrainingMethod> ExpectedOutputs(params double[] outputs)
        {
            this.outputs = outputs; 
            return this; 
        }

        public TrainingEpochTester<TTrainingMethod> ErrorRate(double errorRate)
        {
            this.errorRate = errorRate; 
            return this; 
        }

        public TrainingEpochTester<TTrainingMethod> ExpectNeuralNetworkState(Action<NeuralNetworkAssertor.Builder> action)
        {
            var builder = new NeuralNetworkAssertor.Builder(); 
            action.Invoke(builder);
            neuralNetworkAssertor = builder.Build(); 
            return this; 
        }

        public void PerformEpochAndAssert() 
        {
            var trainingData = TrainingDataSet.For(inputs, outputs); 
            var errorRate = trainingMethod.PerformSingleEpochProducingErrorRate(trainingData);
            AssertResultOfTrainingEpoch(errorRate); 
        }

        private void AssertResultOfTrainingEpoch(double actualErrorRate)
        {
            neuralNetworkAssertor.Assert(trainingMethod.NetworkUnderTraining); 
            Xunit.Assert.Equal(actualErrorRate, errorRate);
        }

        public static TrainingEpochTester<TTrainingMethod> For(TTrainingMethod trainingMethod)
            => new TrainingEpochTester<TTrainingMethod>(trainingMethod); 
    }
}