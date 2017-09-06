using System;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Extensions;
using NeuralNetworks.Library.Training;
using NeuralNetworks.Library.Training.BackPropagation;

namespace NeuralNetworks.Examples.FraudDetection.Services
{
    public class NeuralNetworkConfiguration
    {
        public TrainingController<BackPropagation> NeuralNetworkTrainer => 
            TrainingController.For(
                BackPropagation.WithConfiguration(
                    fraudDetectionNeuralNetwork, 
                    ParallelOptionsExtensions.MultiThreadedOptions(30))); 

        private readonly NeuralNetwork fraudDetectionNeuralNetwork = 
            NeuralNetwork.For(NeuralNetworkContext.MaximumPrecision)
                .WithInputLayer(neuronCount: 30, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 50, activationType: ActivationType.TanH)
                .WithHiddenLayer(neuronCount: 50, activationType: ActivationType.TanH)
                .WithHiddenLayer(neuronCount: 50, activationType: ActivationType.TanH)
                .WithOutputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
                .Build(); 
    }
}