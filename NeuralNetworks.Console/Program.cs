using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Data;
using NeuralNetworks.Library.Extensions;
using NeuralNetworks.Library.Logging;
using NeuralNetworks.Library.Training;
using NeuralNetworks.Library.Training.BackPropagation;

namespace NeuralNetworks.Console
{
    public class Program
    {
        public static void Main(string[] args)
        {
            ConfigureLogging();

            var neuralNetwork = NeuralNetwork.For(NeuralNetworkContext.MaximumPrecision)
                .WithInputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 5, activationType: ActivationType.TanH)
                .WithOutputLayer(neuronCount: 1, activationType: ActivationType.Sigmoid)
                .Build();

            TrainingController
                .For(BackPropagation.WithConfiguration(
                        neuralNetwork, 
                        ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions,
                        learningRate: 0.6, 
                        momentum: 0.9))
                .TrainForEpochsOrErrorThresholdMet(GetXorTrainingData(), maximumEpochs: 5000, errorThreshold: 0.001)
                .GetAwaiter()
                .GetResult();

            MakeExamplePredictions(neuralNetwork);
        }

        private static void ConfigureLogging()
        {
            var logger = new LoggerFactory();

            logger
                .AddConsole(LogLevel.Information)
                .InitialiseLoggingForNeuralNetworksLibrary();
        }

        private static void MakeExamplePredictions(NeuralNetwork neuralNetwork)
        {
            System.Console.WriteLine(
                $"PREDICTION (0, 1): {neuralNetwork.PredictionFor(new[] {0.0, 1.0}, ParallelOptionsExtensions.SingleThreadedOptions)[0]}, EXPECTED: 1");
            System.Console.WriteLine(
                $"PREDICTION (1, 0): {neuralNetwork.PredictionFor(new[] {1.0, 0.0}, ParallelOptionsExtensions.SingleThreadedOptions)[0]}, EXPECTED: 1");
            System.Console.WriteLine(
                $"PREDICTION (0, 0): {neuralNetwork.PredictionFor(new[] {0.0, 0.0}, ParallelOptionsExtensions.SingleThreadedOptions)[0]}, EXPECTED: 0");
            System.Console.WriteLine(
                $"PREDICTION (1, 1): {neuralNetwork.PredictionFor(new[] {1.0, 1.0}, ParallelOptionsExtensions.SingleThreadedOptions)[0]}, EXPECTED: 0");

            if (Debugger.IsAttached) System.Console.ReadLine();
        }

        private static List<TrainingDataSet> GetXorTrainingData()
        {
            var inputs = new[]
            {
                new[] {0.0, 0.0}, new[] {0.0, 1.0}, new[] {1.0, 0.0}, new[] {1.0, 1.0}
            };

            var outputs = new[]
            {
                new[] {0.0}, new[] {1.0}, new[] {1.0}, new[] {0.0}
            };

            return inputs.Select((input, i) => TrainingDataSet.For(input, outputs[i])).ToList();
        }
    }
}