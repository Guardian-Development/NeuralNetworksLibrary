using System;
using System.Linq;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Extensions;
using NeuralNetworks.Library.Training;
using NeuralNetworks.Library.Training.BackPropagation;
using NeuralNetworks.Tests.Support;
using Xunit;
using static System.FormattableString;

namespace NeuralNetworks.Tests.IntegrationTests.DatasetCaseStudies.IrisDatasetCaseStudy
{
    public sealed class IrisDatasetCaseStudy : NeuralNetworkTest
    {
        [Fact(Skip = "Not completed")]
        public async void CanSuccessfullySolveIrisProblemTrainingForEpochsOrErrorThresholdMet()
        {
            var neuralNetwork = NeuralNetwork.For(NeuralNetworkContext.MaximumPrecision)
                .WithInputLayer(neuronCount: 4, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 7, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 3, activationType: ActivationType.Sigmoid)
                .WithOutputLayer(neuronCount: 3, activationType: ActivationType.TanH)
                .Build();

            await TrainingController
                    .For(BackPropagation.WithConfiguration(
                        neuralNetwork,  
                        ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions(),
                        learningRate: 1.15, 
                        momentum: 0.9))
                    .TrainForEpochsOrErrorThresholdMet(IrisDataSet.TrainingData, maximumEpochs: 20000, errorThreshold: 0.01);

            AssertPredictionsForTrainedNeuralNetwork(neuralNetwork);
        }

        private void AssertPredictionsForTrainedNeuralNetwork(NeuralNetwork neuralNetwork)
        {
            IrisDataSet.RowsToBeUsedForPredictions.ForEach(row => 
            {
                var predictions = neuralNetwork.PredictionFor(
                    row.PredictionDataPoints, 
                    ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions());
                
                var prediction = Array.IndexOf(predictions, predictions.Max()); 

                Assert.True(prediction == row.Species, 
                    Invariant($"Row with Id: {row.Id}. Predicted: {prediction}. Actual: {row.Species}")); 
            }); 
        }
    }
}