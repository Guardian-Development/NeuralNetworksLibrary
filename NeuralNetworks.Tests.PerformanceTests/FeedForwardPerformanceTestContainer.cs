using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Attributes.Exporters;
using BenchmarkDotNet.Attributes.Jobs;
using BenchmarkDotNet.Engines;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Extensions;

namespace NeuralNetworks.Tests.PerformanceTests
{
    [MarkdownExporter]
    [SimpleJob(RunStrategy.Monitoring, launchCount: 10, warmupCount: 5, targetCount: 20)]
    public class FeedForwardPerformanceComparisonContainer
    {
        private readonly NeuralNetwork neuralNetworkUnderTest;

        public FeedForwardPerformanceComparisonContainer()
        {
            neuralNetworkUnderTest = 
                NeuralNetwork
                    .For(NeuralNetworkContext.MaximumPrecision)
                    .WithInputLayer(neuronCount: 5, activationType: ActivationType.Sigmoid)
                    .WithHiddenLayer(neuronCount: 100, activationType: ActivationType.Sigmoid)
                    .WithHiddenLayer(neuronCount: 100, activationType: ActivationType.TanH)
                    .WithHiddenLayer(neuronCount: 100, activationType: ActivationType.TanH)
                    .WithHiddenLayer(neuronCount: 100, activationType: ActivationType.Sigmoid)
                    .WithHiddenLayer(neuronCount: 100, activationType: ActivationType.TanH)
                    .WithHiddenLayer(neuronCount: 100, activationType: ActivationType.Sigmoid)
                    .WithOutputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
                    .Build();
        }

        private double[] InputValues => new [] { 0.98, 0.23, 0.44, 0.44, 0.12 }; 

        [Benchmark]
        public double[] FeedForwardPredictionSingleThreaded() => 
            neuralNetworkUnderTest.PredictionFor(InputValues, ParallelOptionsExtensions.SingleThreadedOptions); 

        [Benchmark]
        public double[] FeedForwardPredictionMultiThreaded() => 
            neuralNetworkUnderTest.PredictionFor(InputValues, ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions); 
    }
}