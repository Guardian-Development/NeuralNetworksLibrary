using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Data;
using NeuralNetworks.Library.Extensions;
using NeuralNetworks.Library.Training;
using NeuralNetworks.Library.Training.BackPropagation;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.DatasetCaseStudies.XorDatasetCaseStudy
{
    public sealed class XorDatasetCaseStudy : NeuralNetworkTest
    {

        [Fact]
        public async void CanSuccessfullySolveXorProblemTrainingForEpochsOrErrorThresholdMet()
        {
            var neuralNetwork = NeuralNetwork.For(NeuralNetworkContext.MaximumPrecision)
                .WithInputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 2, activationType: ActivationType.TanH)
                .WithOutputLayer(neuronCount: 1, activationType: ActivationType.Sigmoid)
                .Build();

            await TrainingController
                    .For(BackPropagation.WithConfiguration(
                        neuralNetwork,  
                        ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions,
                        learningRate: 0.4, 
                        momentum: 0.9))
                    .TrainForEpochsOrErrorThresholdMet(XorTrainingData(), maximumEpochs: 3000, errorThreshold: 0.01);

            AssertPredictionsForTrainedNeuralNetwork(neuralNetwork); 
        }

        [Fact]
        public async void CanSuccessfullySolveXorProblemTrainingForEpochs()
        {
            var neuralNetwork = NeuralNetwork.For(NeuralNetworkContext.MaximumPrecision)
                .WithInputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 2, activationType: ActivationType.TanH)
                .WithOutputLayer(neuronCount: 1, activationType: ActivationType.Sigmoid)
                .Build();

            await TrainingController
                    .For(BackPropagation.WithConfiguration(
                        neuralNetwork,  
                        ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions,
                        learningRate: 0.4, 
                        momentum: 0.9))
                    .TrainForEpochs(XorTrainingData(), maximumEpochs: 5000);

            AssertPredictionsForTrainedNeuralNetwork(neuralNetwork);
        }

        [Fact]
        public async void CanSuccessfullySolveXorProblemTrainingForErrorThreshold()
        {
            var neuralNetwork = NeuralNetwork.For(NeuralNetworkContext.MaximumPrecision)
                .WithInputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 2, activationType: ActivationType.TanH)
                .WithOutputLayer(neuronCount: 1, activationType: ActivationType.Sigmoid)
                .Build();

            await TrainingController
                    .For(BackPropagation.WithConfiguration(
                        neuralNetwork,  
                        ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions,
                        learningRate: 0.4, 
                        momentum: 0.9))
                    .TrainForErrorThreshold(XorTrainingData(), minimumErrorThreshold: 0.01);

            AssertPredictionsForTrainedNeuralNetwork(neuralNetwork); 
        }

        private void AssertPredictionsForTrainedNeuralNetwork(NeuralNetwork neuralNetwork)
        {
            Assert.True(neuralNetwork.PredictionFor(new [] { 0.0, 1.0 }, 
                ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions)[0] >= 0.5,
                    "Prediction incorrect for (0, 1)");
            Assert.True(neuralNetwork.PredictionFor(new [] { 1.0, 0.0 }, 
                ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions)[0] >= 0.5, 
                    "Prediction incorrect for (1, 0)");
            Assert.True(neuralNetwork.PredictionFor(new[] { 0.0, 0.0 }, 
                ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions)[0] < 0.5, 
                    "Prediction incorrect for (0, 0)");
            Assert.True(neuralNetwork.PredictionFor(new[] { 1.0, 1.0 }, 
                ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions)[0] < 0.5, 
                    "Prediction incorrect for (1, 1)");
        }

        private static List<TrainingDataSet> XorTrainingData()
        {
            var inputs = new[]
            {
                new[] {0.0, 0.0}, new[] {0.0, 1.0}, new[] {1.0, 0.0}, new[] {1.0, 1.0}
            };

            var outputs = new[]
            {
                new[] {0.0}, new[] {1.0}, new[] {1.0}, new[] {0.0}
            };

            return TrainingDataSetExtensions.BuildTrainingDataForAllInputs(inputs, outputs).ToList(); 
        }
    }
}
