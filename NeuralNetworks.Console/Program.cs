using Microsoft.Extensions.Logging;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Logging;
using NeuralNetworks.Library.Training;

namespace NeuralNetworks.Console
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var logger = new LoggerFactory();

            logger
                .AddConsole(LogLevel.Information)
                .InitialiseLoggingForNeuralNetworksLibrary();

            var neuralNetwork = NeuralNetwork.For()
                .WithInputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 3, activationType: ActivationType.Sigmoid)
                .WithOutputLayer(neuronCount: 1, activationType: ActivationType.Sigmoid)
                .Build();

            GetXorTrainingData(out var trainingInputs, out var trainingOutputs);

            BackPropagation
                .For(neuralNetwork, learningRate: 0.1, momentum: 0.9)
                .TrainNetwork(trainingInputs, trainingOutputs, epochs: 1000);

            System.Console.WriteLine(
                $"PREDICTION (0, 1): {neuralNetwork.MakePrediction(new[] { 0.0, 1.0 })[0]}, EXPECTED: 1");
            System.Console.WriteLine(
                $"PREDICTION (1, 0): {neuralNetwork.MakePrediction(new[] { 0.0, 1.0 })[0]}, EXPECTED: 1");
            System.Console.WriteLine(
                $"PREDICTION (0, 0): {neuralNetwork.MakePrediction(new[] { 0.0, 1.0 })[0]}, EXPECTED: 0");
            System.Console.WriteLine(
                $"PREDICTION (1, 1): {neuralNetwork.MakePrediction(new[] { 0.0, 1.0 })[0]}, EXPECTED: 0");

            if (System.Diagnostics.Debugger.IsAttached) System.Console.ReadLine();
        }

        private static void GetXorTrainingData(
            out double[][] trainingInputs,
            out double[][] trainingOutputs)
        {
            trainingInputs = new[]
            {
                new[] {0.0, 0.0}, new[] {0.0, 1.0}, new[] {1.0, 0.0}, new[] {1.0, 1.0}
            };

            trainingOutputs = new[]
            {
                new[] {0.0}, new[] {1.0}, new[] {1.0}, new[] {0.0}
            };
        }
    }
}