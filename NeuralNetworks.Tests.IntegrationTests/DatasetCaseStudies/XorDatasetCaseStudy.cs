using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Data;
using NeuralNetworks.Library.Training;
using NeuralNetworks.Library.Training.BackPropagation;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.DatasetCaseStudies
{
    public sealed class XorDatasetCaseStudy : NeuralNetworkTest
    {
        [Fact]
        public void CanSuccessfullySolveXorProblem()
        {
            var neuralNetwork = NeuralNetwork.For(NeuralNetworkContext.MaximumPrecision)
                .WithInputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 2, activationType: ActivationType.TanH)
                .WithOutputLayer(neuronCount: 1, activationType: ActivationType.Sigmoid)
                .Build();

            TrainingController
                .For(BackPropagation.WithMultiThreadedConfiguration(
                    neuralNetwork,  
                    new ParallelOptions(),
                    learningRate: 0.4, 
                    momentum: 0.9))
                .TrainForEpochsOrErrorThresholdMet(XorTrainingData(), maximumEpochs: 3000, errorThreshold: 0.01);

            Assert.True(neuralNetwork.PredictionFor(0.0, 1.0)[0] >= 0.5, "Prediction incorrect for (0, 1)");
            Assert.True(neuralNetwork.PredictionFor(1.0, 0.0)[0] >= 0.5, "Prediction incorrect for (1, 0)");
            Assert.True(neuralNetwork.PredictionFor(0.0, 0.0)[0] < 0.5, "Prediction incorrect for (0, 0)");
            Assert.True(neuralNetwork.PredictionFor(1.0, 1.0)[0] < 0.5, "Prediction incorrect for (1, 1)");
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

            return inputs.Zip(outputs, (input, output) => TrainingDataSet.For(input, output))
                         .ToList(); 
        }
    }
}
