using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Examples.FraudDetection.Services.Domain;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Logging;
using NeuralNetworks.Library.Training;

namespace NeuralNetworks.Examples.FraudDetection.Services
{
    public static class ServiceLayerConfiguration
    {
        public static IServiceCollection ConfigureServiceLayer(this IServiceCollection services)
        {
            return services
                .AddSingleton(new NeuralNetworkAccessor(NeuralNetwork))
                .AddSingleton(typeof(DataProvider));
        }

        private static NeuralNetwork NeuralNetwork => 
            NeuralNetwork.For(NeuralNetworkContext.MaximumPrecision)
                .WithInputLayer(neuronCount: 29, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 18, activationType: ActivationType.TanH)
                .WithHiddenLayer(neuronCount: 27, activationType: ActivationType.TanH)
                .WithHiddenLayer(neuronCount: 40, activationType: ActivationType.TanH)
                .WithOutputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
                .Build(); 
        
        public static ILoggerFactory ConfigureServiceLayerLogging(this ILoggerFactory loggerFactory)
        {
            return loggerFactory.InitialiseLoggingForNeuralNetworksLibrary(); 
        }
    }
}