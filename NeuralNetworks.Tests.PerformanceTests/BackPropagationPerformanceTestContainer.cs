using System.Collections.Generic;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Data;
using NeuralNetworks.Library.Extensions;
using NeuralNetworks.Library.Training;
using NeuralNetworks.Library.Training.BackPropagation;

namespace NeuralNetworks.Tests.PerformanceTests
{
    public class BackPropagationPerformanceComparisonContainer
    {
        private readonly TrainingController<BackPropagation> singleThreadedController; 
        private readonly TrainingController<BackPropagation> multiThreadedController; 
        private readonly IList<TrainingDataSet> trainingData;

        public BackPropagationPerformanceComparisonContainer()
        {
            var neuralNetworkUnderTest = 
                NeuralNetwork
                    .For(NeuralNetworkContext.MaximumPrecision)
                    .WithInputLayer(neuronCount: 5, activationType: ActivationType.Sigmoid)
                    .WithHiddenLayer(neuronCount: 100, activationType: ActivationType.Sigmoid)
                    .WithHiddenLayer(neuronCount: 70, activationType: ActivationType.TanH)
                    .WithHiddenLayer(neuronCount: 40, activationType: ActivationType.TanH)
                    .WithHiddenLayer(neuronCount: 100, activationType: ActivationType.Sigmoid)
                    .WithOutputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
                    .Build();

            trainingData = new[] { 
                TrainingDataSet.For(new [] {0.78, 0.99, 0.67, 0.72, 0.22}, new [] { 0.12, 0.14 })}; 

            multiThreadedController = TrainingController.For(
                BackPropagation
                    .WithConfiguration(neuralNetworkUnderTest, ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions()));

            singleThreadedController = TrainingController.For(
                BackPropagation
                    .WithConfiguration(neuralNetworkUnderTest, ParallelOptionsExtensions.SingleThreadedOptions()));
        }

        [Benchmark]
        public void BackPropagationTrainingMultiThreadedTenThousandEpochs()
            => multiThreadedController.TrainForEpochs(trainingData, maximumEpochs: 10000).GetAwaiter().GetResult();

        [Benchmark]
        public void BackPropagationTrainingSingleThreadedTenThousandEpochs()
            => singleThreadedController.TrainForEpochs(trainingData, maximumEpochs: 10000).GetAwaiter().GetResult();
    }
}