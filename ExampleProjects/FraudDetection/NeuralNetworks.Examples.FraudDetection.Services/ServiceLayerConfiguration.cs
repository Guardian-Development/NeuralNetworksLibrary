using Microsoft.Extensions.DependencyInjection;
using NeuralNetworks.Examples.FraudDetection.Services.Domain;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Training;

namespace NeuralNetworks.Examples.FraudDetection.Services
{
    public static class ServiceLayerConfiguration
    {
        public static IServiceCollection ConfigureServiceLayer(this IServiceCollection services)
        {
            return services
                .AddSingleton(new NeuralNetworkAccessor(NeuralNetwork))
                .AddTransient(typeof(TrainingDataProvider));
        }

        private static NeuralNetwork NeuralNetwork => 
            NeuralNetwork.For(NeuralNetworkContext.MaximumPrecision)
                .WithInputLayer(neuronCount: 30, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 40, activationType: ActivationType.TanH)
                .WithHiddenLayer(neuronCount: 40, activationType: ActivationType.TanH)
                .WithHiddenLayer(neuronCount: 40, activationType: ActivationType.TanH)
                .WithHiddenLayer(neuronCount: 40, activationType: ActivationType.TanH)
                .WithOutputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
                .Build(); 
    }
}