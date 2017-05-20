using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;

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
        }
    }
}