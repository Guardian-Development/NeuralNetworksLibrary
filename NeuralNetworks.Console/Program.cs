using System.Runtime.InteropServices;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Training;

namespace NeuralNetworks.Console
{
    public class Program
    {
        public static void Main(string[] args)
        {
            System.Console.WriteLine("Creating Neural Network");

            var neuralNetwork = NeuralNetwork.Create()
                .WithInputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 3, activationType: ActivationType.Sigmoid)
                .WithOutputLayer(neuronCount: 1, activationType: ActivationType.Sigmoid);

            System.Console.WriteLine("Creating Training Data");

            GetXorTrainingData(out var trainingInputs, out var trainingOutputs);

            System.Console.WriteLine("Training Neural Network");

            BackPropagation
                .For(neuralNetwork, learningRate: 0.1, momentum: 0.9)
                .TrainNetwork(trainingInputs, trainingOutputs); 
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