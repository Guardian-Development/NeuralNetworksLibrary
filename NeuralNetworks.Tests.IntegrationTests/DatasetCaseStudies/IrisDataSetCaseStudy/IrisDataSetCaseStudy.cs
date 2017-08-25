using System;
using System.Collections.Generic;
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
        [Fact]
        public async void CanSuccessfullySolveIrisProblemTrainingForEpochsOrErrorThresholdMet()
        {
            var neuralNetwork = NeuralNetwork.For(NeuralNetworkContext.MaximumPrecision)
                .WithInputLayer(neuronCount: 4, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 8, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 5, activationType: ActivationType.Sigmoid)
                .WithOutputLayer(neuronCount: 3, activationType: ActivationType.TanH)
                .Build();

            await TrainingController
                    .For(BackPropagation.WithConfiguration(
                        neuralNetwork,  
                        ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions,
                        learningRate: 1.15, 
                        momentum: 0.4))
                    .TrainForEpochsOrErrorThresholdMet(IrisDataSet.TrainingData, maximumEpochs: 1500, errorThreshold: 0.01);

            AssertPredictionsForTrainedNeuralNetwork(neuralNetwork);
        }

        [Fact]
        public async void CanSuccessfullySolveIrisProblemTrainingForEpochs()
        {
            var neuralNetwork = NeuralNetwork.For(NeuralNetworkContext.MaximumPrecision)
                .WithInputLayer(neuronCount: 4, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 8, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 5, activationType: ActivationType.Sigmoid)
                .WithOutputLayer(neuronCount: 3, activationType: ActivationType.TanH)
                .Build();

            await TrainingController
                    .For(BackPropagation.WithConfiguration(
                        neuralNetwork,  
                        ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions,
                        learningRate: 1.15, 
                        momentum: 0.4))
                    .TrainForEpochs(IrisDataSet.TrainingData, maximumEpochs: 1000);

            AssertPredictionsForTrainedNeuralNetwork(neuralNetwork);
        }

        [Fact]
        public async void CanSuccessfullySolveXorProblemTrainingForErrorThreshold()
        {
            var neuralNetwork = NeuralNetwork.For(NeuralNetworkContext.MaximumPrecision)
                .WithInputLayer(neuronCount: 4, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 8, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 5, activationType: ActivationType.Sigmoid)
                .WithOutputLayer(neuronCount: 3, activationType: ActivationType.TanH)
                .Build();

            await TrainingController
                    .For(BackPropagation.WithConfiguration(
                        neuralNetwork,  
                        ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions,
                        learningRate: 1.15, 
                        momentum: 0.4))
                    .TrainForErrorThreshold(IrisDataSet.TrainingData, minimumErrorThreshold: 0.08);

            AssertPredictionsForTrainedNeuralNetwork(neuralNetwork);
        }

        private void AssertPredictionsForTrainedNeuralNetwork(NeuralNetwork neuralNetwork)
        {
            var predictedCorrectly = new List<IrisDataRow>(); 
            var predictedIncorrectly = new List<IrisDataRow>(); 

            IrisDataSet.RowsToBeUsedForPredictions.ForEach(row => 
            {
                var predictions = neuralNetwork.PredictionFor(
                    row.PredictionDataPoints, 
                    ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions);
                
                var prediction = Array.IndexOf(predictions, predictions.Max()) + 1; 

                if(prediction == row.Species) 
                {
                    predictedCorrectly.Add(row); 
                }
                else 
                {
                    predictedIncorrectly.Add(row); 
                }
            }); 

            Assert.True(predictedIncorrectly.Count == 0, 
                    Invariant($"Predicted Incorrectly Count: {predictedIncorrectly.Count}. Predicted Correctly Count: {predictedCorrectly.Count}")); 
        }
    }
}